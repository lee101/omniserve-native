#!/usr/bin/env python3
"""ra2 (Qwen Image 2.1) residency / cache A/B on the shared 5090.

Each arm = live omniserve-native-qwen.service env + overrides, served by a throwaway gateway on --port
(default 8794). Same seeds in every arm. Per request: wall, inference, TE / sampling / VAE seconds from the
sd.cpp INFO log, peak process VRAM (NVML poll). Aborts on >= --abort-5xx authenticated prod 5xx.

uv run --with requests --with pillow --with numpy tools/ra2_residency_bench.py --arms arms.json --out DIR
arms.json: {"prod": {"env": {}, "need_mib": 6000}, "resident": {"env": {"OMNISERVE_NATIVE_SD_PARAMS_BACKEND": ""}, "need_mib": 14000}}
"""
import argparse, base64, json, os, re, shlex, signal, statistics, subprocess, threading, time
from pathlib import Path

import requests

UNIT = "omniserve-native-qwen.service"
PROMPTS = ["a red fox reading a newspaper in a cafe, photo",
           "close-up portrait of an elderly woman with freckles and silver hair, soft window light, 85mm photo",
           "a vintage travel poster that says \"VISIT AUCKLAND\" in bold art deco letters, harbour bridge at sunset",
           "isometric fantasy village on a floating island, waterfalls, detailed digital painting"]
EDIT_PROMPTS = ["make it a snowy winter scene", "turn it into a watercolor painting",
                "add a small red umbrella", "change the time of day to night"]
CANARY_ENV = {"OMNISERVE_ACCESS_LOG": "0", "OMNISERVE_NATIVE_GUARD": "0", "OMNISERVE_NATIVE_GUARD_JUDGE": "0",
              "OMNISERVE_NATIVE_SECRET": "", "OMNISERVE_NATIVE_IMAGE_OVERFLOW_UPSTREAM": "",
              "OMNISERVE_NATIVE_FRONTIER_POLICY": "", "OMNISERVE_NATIVE_SD_LOG_LEVEL": "2",
              "OMNISERVE_NATIVE_VRAM_OWNER": "qwen-canary", "OMNISERVE_NATIVE_VRAM_FORCE_TIERS": "",
              "OMNISERVE_NATIVE_RAM_PREFETCH_ENABLED": "0"}
TIMING = {"te": re.compile(r"get_learned_condition completed, taking ([\d.]+)s"),
          "sampling": re.compile(r"sampling completed, taking ([\d.]+)s"),
          "vae": re.compile(r"decode_first_stage completed, taking ([\d.]+)s"),
          "encode": re.compile(r"encode_first_stage completed, taking ([\d.]+)s"),
          "lora": re.compile(r"apply_loras completed, taking ([\d.]+)s"),
          "total": re.compile(r"generate_image completed in ([\d.]+)s")}


def unit_env():
    raw = subprocess.run(["systemctl", "show", UNIT, "-p", "Environment", "--value"], capture_output=True, text=True).stdout
    env = dict(kv.split("=", 1) for kv in shlex.split(raw) if "=" in kv)
    exe = subprocess.run(["systemctl", "show", UNIT, "-p", "ExecStart", "--value"], capture_output=True, text=True).stdout
    binary = exe.split("path=")[1].split(";")[0].strip()
    wd = subprocess.run(["systemctl", "show", UNIT, "-p", "WorkingDirectory", "--value"], capture_output=True, text=True).stdout.strip()
    return env, binary, wd


def smi(query):
    return subprocess.run(["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout


def free_mib():
    return int(smi("gpu=memory.free").split()[0])


def pid_mib(pid):
    for line in smi("compute-apps=pid,used_memory").splitlines():
        p, _, m = line.partition(",")
        if p.strip() == str(pid):
            return int(m)
    return 0


class Sampler(threading.Thread):
    """Polls the canary's VRAM; kills it if device free drops under floor_mib (protects prod tenants)."""
    def __init__(self, pid, floor_mib=0):
        super().__init__(daemon=True)
        self.pid, self.peak, self.min_free, self.stop_ev = pid, 0, 1 << 30, threading.Event()
        self.floor, self.tripped = floor_mib, False

    def run(self):
        while not self.stop_ev.is_set():
            self.peak = max(self.peak, pid_mib(self.pid))
            f = free_mib()
            self.min_free = min(self.min_free, f)
            if self.floor and f < self.floor and not self.tripped:
                self.tripped = True
                print(json.dumps({"vram_floor_trip": f, "canary_mib": pid_mib(self.pid)}), flush=True)
                try:
                    os.killpg(self.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            self.stop_ev.wait(0.1)

    def take(self):
        p, f = self.peak, self.min_free
        self.peak, self.min_free = pid_mib(self.pid), 1 << 30
        return p, f


class ProdWatch(threading.Thread):
    """Counts authenticated prod 5xx in the shared access log since start."""
    def __init__(self, path="/var/log/omniserve/access.log"):
        super().__init__(daemon=True)
        self.path, self.count, self.lines = path, 0, []
        self.start_ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())

    def run(self):
        with open(self.path) as f:
            f.seek(0, 2)
            while True:
                line = f.readline()
                if not line:
                    time.sleep(1)
                    continue
                if re.search(r" status=5\d\d ", line) and re.search(r"hdrs=[^-]", line) and "internal=1" not in line:
                    self.count += 1
                    self.lines.append(line.strip())


def port_busy(port):
    return f":{port} " in subprocess.run(["ss", "-ltnH"], capture_output=True, text=True).stdout


def start(binary, wd, env, port, log):
    if port_busy(port):
        raise RuntimeError(f"port {port} already has a listener")
    p = subprocess.Popen([binary, "--port", str(port)], cwd=wd, env={**os.environ, **env},
                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 600:
        if p.poll() is not None:
            raise RuntimeError(f"server exited rc={p.returncode}")
        try:
            if requests.get(f"http://127.0.0.1:{port}/status", timeout=2).ok:
                return p, time.monotonic() - t0
        except requests.RequestException:
            pass
        time.sleep(0.5)
    stop(p)
    raise RuntimeError("server not ready")


def stop(p):
    try:
        os.killpg(p.pid, signal.SIGTERM)
        p.wait(30)
    except Exception:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def gen(port, case, prompt, seed, src_b64, png):
    body = {"prompt": prompt, "width": case.get("w", 1024), "height": case.get("h", 1024), "seed": seed,
            "cache": False, "output_format": "png", **case.get("body", {})}
    path = "generations"
    if case["kind"] == "edit":
        body["image_base64"] = src_b64
        path = "edits"
    for attempt in range(6):
        t0 = time.monotonic()
        r = requests.post(f"http://127.0.0.1:{port}/v1/images/{path}", json=body, timeout=900,
                          headers={"X-Omniserve-Tier": "background"})
        wall = time.monotonic() - t0
        if r.status_code == 503:
            time.sleep(30)
            continue
        r.raise_for_status()
        d = r.json()["data"][0]
        png.write_bytes(base64.b64decode(d["b64_json"]))
        return {"wall": round(wall, 3), "inference_ms": d.get("inference_time_ms"),
                "denoiser_cache": d.get("denoiser_cache"), "retries": attempt}
    raise RuntimeError("canary kept returning 503 (VRAM lease denied)")


def timings(log_path, offset):
    with open(log_path, errors="replace") as f:
        f.seek(offset)
        text = f.read()
    out = {}
    for k, rx in TIMING.items():
        m = rx.findall(text)
        if m:
            out[k] = float(m[-1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    ap.add_argument("--cases", default="[]", help='JSON list: [{"name":"t2i","kind":"t2i","steps":20,"body":{}}]')
    ap.add_argument("--exec", dest="exec_cmd", help="shell command run against each arm's canary (env CANARY_BASE, ARM, ARM_DIR)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=8794)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--per-case", type=int, default=2)
    ap.add_argument("--edit-src", default="/nvme0n1-disk/code/omniserve-native-ra2overflow/experiments/qwen21-speed/canary_final_t2i/prod/p0_r0_i0.png")
    ap.add_argument("--margin-mib", type=int, default=4096)
    ap.add_argument("--free-wait-s", type=int, default=1800)
    ap.add_argument("--abort-5xx", type=int, default=3)
    ap.add_argument("--floor-mib", type=int, default=2048, help="kill the canary if device free falls below this")
    a = ap.parse_args()
    if a.port in (8790, 8791, 8792, 8799):
        raise SystemExit("refusing a prod port")
    arms = json.loads(Path(a.arms).read_text())
    cases = json.loads(Path(a.cases).read_text()) if a.cases.endswith(".json") else json.loads(a.cases)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    src_b64 = base64.b64encode(Path(a.edit_src).read_bytes()).decode()
    base, binary, wd = unit_env()
    watch = ProdWatch(); watch.start()
    runs = out / "runs.jsonl"
    names = list(arms)
    for rnd in range(a.rounds):
        order = names[rnd % len(names):] + names[:rnd % len(names)]
        for name in order:
            arm = arms[name]
            need = arm.get("need_mib", 8000) + a.margin_mib
            t0 = time.monotonic()
            while (f := free_mib()) < need:
                if time.monotonic() - t0 > a.free_wait_s:
                    print(json.dumps({"arm": name, "skipped": f"free {f} < {need} MiB"}), flush=True)
                    with runs.open("a") as fh:
                        fh.write(json.dumps({"arm": name, "round": rnd, "skipped": f"free {f} < {need}"}) + "\n")
                    break
                time.sleep(20)
            else:
                d = out / name; d.mkdir(exist_ok=True)
                log_path = d / f"server_r{rnd}.log"
                env = {**base, **CANARY_ENV, "OMNISERVE_NATIVE_PORT": str(a.port),
                       "OMNISERVE_NATIVE_FRONTIER_LOG": str(out / "frontier.jsonl"), **arm.get("env", {})}
                with open(log_path, "w") as log:
                    p, ready_s = start(env.pop("OMNISERVE_CANARY_BINARY", binary), wd, env, a.port, log)
                    s = Sampler(p.pid, a.floor_mib); s.start()
                    try:
                        print(json.dumps({"arm": name, "ready_s": round(ready_s, 1), "idle_mib": pid_mib(p.pid)}), flush=True)
                        if a.exec_cmd:
                            rc = subprocess.run(a.exec_cmd, shell=True, env={**os.environ, "CANARY_BASE": f"http://127.0.0.1:{a.port}",
                                                "ARM": name, "ARM_DIR": str(d)}).returncode
                            peak, mfree = s.take()
                            with runs.open("a") as fh:
                                fh.write(json.dumps({"arm": name, "round": rnd, "exec_rc": rc, "peak_mib": peak, "min_free_mib": mfree}) + "\n")
                            print(json.dumps({"arm": name, "exec_rc": rc, "peak_mib": peak, "min_free_mib": mfree, "prod_5xx": watch.count}), flush=True)
                        for case in cases:
                            prompts = EDIT_PROMPTS if case["kind"] == "edit" else PROMPTS
                            seq = [("warm", 1)] + [(i, 1000 + i) for i in range(a.per_case)]
                            for i, seed in seq:
                                if s.tripped:
                                    raise SystemExit("ABORT: device free fell under the VRAM floor")
                                if watch.count >= a.abort_5xx:
                                    raise SystemExit(f"ABORT prod 5xx={watch.count}: {watch.lines[-3:]}")
                                pi = 0 if i == "warm" else i % len(prompts)
                                case_body = {**case, "body": {**({"steps": case["steps"]} if "steps" in case else {}), **case.get("body", {})}}
                                off = log_path.stat().st_size
                                s.take()
                                r = gen(a.port, case_body, prompts[pi], seed, src_b64, d / f"{case['name']}_{i}_r{rnd}.png")
                                time.sleep(0.3)
                                peak, mfree = s.take()
                                r.update(arm=name, case=case["name"], round=rnd, i=i, seed=seed, prompt=pi,
                                         peak_mib=peak, min_free_mib=mfree, **timings(log_path, off))
                                with runs.open("a") as fh:
                                    fh.write(json.dumps(r) + "\n")
                                print(json.dumps(r), flush=True)
                    finally:
                        s.stop_ev.set()
                        stop(p)
    summarize(runs)
    print(f"prod authenticated 5xx during run: {watch.count}")


def summarize(runs):
    rows = [json.loads(l) for l in runs.read_text().splitlines()]
    rows = [r for r in rows if "case" in r and r.get("i") != "warm"]
    keys = sorted({(r["arm"], r["case"]) for r in rows})
    print(f"\n{'arm':22s} {'case':10s} {'n':>2s} {'wall_med':>8s} {'te':>6s} {'samp':>6s} {'s/it':>6s} {'vae':>6s} {'peak':>6s}")
    for arm, case in keys:
        rs = [r for r in rows if r["arm"] == arm and r["case"] == case]
        med = lambda k: statistics.median([r[k] for r in rs if k in r]) if any(k in r for r in rs) else float("nan")
        print(f"{arm:22s} {case:10s} {len(rs):2d} {med('wall'):8.2f} {med('te'):6.2f} {med('sampling'):6.2f} "
              f"{'':>6s} {med('vae'):6.2f} {max(r['peak_mib'] for r in rs):6d}")


if __name__ == "__main__":
    main()
