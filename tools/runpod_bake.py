"""Bake model weights into a worker image in-datacenter (home uplink ~1 MB/s).

usage: python3 tools/runpod_bake.py h3|pixal
Creates a throwaway serverless template+endpoint on a cheap GPU, runs tools/runpod_weight_baker.py
as the entrypoint of the target image, crane-appends one layer per >1 GB file, and deletes the
endpoint and template in a finally block. Builder cost 2026-10-09: ~/bin/bash.15 (pixal, 14 min) / ~/bin/bash.25 (h3, 30 min).
"""
import base64, json, sys, time, urllib.request, urllib.error
import os


def _key():
    if os.environ.get("RUNPOD_API_KEY"):
        return os.environ["RUNPOD_API_KEY"]
    for line in open(os.environ.get("RUNPOD_ENV_FILE", "/nvme0n1-disk/code/manifoldgen-site/.env")):
        if line.startswith(("RUNPOD_API_KEY=", "H3_RUNPOD_API_KEY=")):
            return line.split("=", 1)[1].strip().strip("\"'")
    raise SystemExit("RUNPOD_API_KEY missing")


K = _key()

REST = "https://rest.runpod.io/v1"
Q = "https://api.runpod.ai/v2/"
H = {"Authorization": "Bearer " + K, "Content-Type": "application/json", "User-Agent": "rp-bake/1.0"}


def req(method, url, body=None):
    r = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None, method=method, headers=H)
    try:
        with urllib.request.urlopen(r, timeout=60) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{method} {url} {e.code} {e.read().decode()[:400]}")


def ghcr_token():
    a = json.load(open("/home/administrator/.docker/config.json"))["auths"]["ghcr.io"]["auth"]
    return base64.b64decode(a).decode().split(":", 1)[1]


def source_env(endpoint_id):
    ep = req("GET", f"{REST}/endpoints/{endpoint_id}?includeTemplate=true")
    return ep["template"].get("env") or {}, ep["template"].get("containerRegistryAuthId")


def main(mode, base, target, src_endpoint, cuda, extra_env):
    env, auth = source_env(src_endpoint)
    env = {k: v for k, v in env.items() if not k.startswith(("S3_",))}
    env["BAKE_B64"] = base64.b64encode(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runpod_weight_baker.py"), "rb").read()).decode()
    name = f"bake-{mode}-{int(time.time())}"
    tpl = ep = job = None
    st = {}
    t0 = time.time()
    try:
        tpl = req("POST", f"{REST}/templates", {
            "name": name, "imageName": base, "isServerless": True, "containerDiskInGb": 200,
            "containerRegistryAuthId": auth, "env": env,
            "dockerEntrypoint": ["python", "-c", "import os,base64;exec(base64.b64decode(os.environ['BAKE_B64']))"],
            "dockerStartCmd": []})["id"]
        body = {"name": name, "templateId": tpl, "gpuTypeIds": ["NVIDIA RTX A5000", "NVIDIA GeForce RTX 3090", "NVIDIA RTX A4000",
                "NVIDIA GeForce RTX 4090", "NVIDIA L4", "NVIDIA RTX A6000", "NVIDIA L40S"], "gpuCount": 1, "workersMin": 0,
                "workersMax": 1, "idleTimeout": 5, "executionTimeoutMs": 3600000, "flashboot": False, "scalerType": "QUEUE_DELAY",
                "scalerValue": 4}
        if cuda:
            body["allowedCudaVersions"] = [cuda]
        ep = req("POST", f"{REST}/endpoints", body)["id"]
        print(name, "template", tpl, "endpoint", ep, flush=True)
        job = req("POST", Q + ep + "/run", {"input": {"mode": mode, "base": base, "target": target, "token": ghcr_token(), "env": extra_env},
                                           "policy": {"executionTimeout": 3600000, "ttl": 5400000}})["id"]
        last = None
        while time.time() - t0 < 5000:
            st = req("GET", Q + ep + "/status/" + job)
            if st.get("status") != last:
                print(round(time.time() - t0), st.get("status"), flush=True)
                last = st.get("status")
            if st.get("status") in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
                break
            time.sleep(10)
    finally:
        if ep and job and st.get("status") not in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
            try: req("POST", Q + ep + "/cancel/" + job)
            except Exception as e: print("cancel", e)
        if ep:
            try: req("PATCH", f"{REST}/endpoints/{ep}", {"workersMax": 0, "workersMin": 0})
            except Exception as e: print("scale0", e)
            for _ in range(12):
                try:
                    req("DELETE", f"{REST}/endpoints/{ep}"); print("deleted endpoint", ep); break
                except Exception as e:
                    print("delete endpoint retry", str(e)[:120]); time.sleep(10)
        if tpl:
            for _ in range(6):
                try:
                    req("DELETE", f"{REST}/templates/{tpl}"); print("deleted template", tpl); break
                except Exception as e:
                    print("delete template retry", str(e)[:120]); time.sleep(10)
    out = st.get("output") or {}
    print(json.dumps({"status": st.get("status"), "delay_s": (st.get("delayTime") or 0) / 1000, "exec_s": (st.get("executionTime") or 0) / 1000,
                      "error": st.get("error"), "output": out}, indent=1)[:6000], flush=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "h3":
        main("h3", "ghcr.io/lee101/h3-cog:cu130-20261008-pinkcherry-opt-r3", "ghcr.io/lee101/h3-cog:cu130-20261009-pinkcherry-r3-baked",
             "811c0jb28shcnr", "13.0", [])
    else:
        main("pixal", "ghcr.io/lee101/pixal3dcog:sls", "ghcr.io/lee101/pixal3dcog:sls-baked-20261009", "akgefm0nzzr4jo", None,
             ["PIXAL3D_HF_HOME=/hf"])
