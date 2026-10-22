#!/usr/bin/env python3
"""Greedy parity and task accuracy for the LLM path, sequential vs concurrent.

Runs a fixed prompt set at temperature 0 one at a time, then all at once with
extra filler streams so every prompt shares its decode steps with other
sequences. Writes the outputs as JSON so two servers (legacy contexts vs the
batched engine) can be compared with --against.

  scripts/llm_batch_parity.py --port 8795 --out /tmp/batched.json
  scripts/llm_batch_parity.py --port 8796 --out /tmp/legacy.json --against /tmp/batched.json

MTP exactness: run an MTP-off and an MTP-on server (CPU: kernels are
batch-width invariant there, so any mismatch is a logic bug) and compare:
  scripts/llm_batch_parity.py --port 8797 --no-concurrent --limit 8 --out /tmp/off.json
  scripts/llm_batch_parity.py --port 8798 --no-concurrent --limit 8 --against /tmp/off.json
"""
import argparse
import http.client
import json
import threading

TASKS = [
    ("What is the capital of Australia? Answer in one word.", ["canberra"]),
    ("What is 17 multiplied by 23? Reply with just the number.", ["391"]),
    ("Which planet is known as the Red Planet? One word.", ["mars"]),
    ("Spell the word 'necessary' backwards, letters only.", ["yrassecen"]),
    ("Who wrote the play Romeo and Juliet? Give the surname only.", ["shakespeare"]),
    ("What is the chemical symbol for gold?", ["au"]),
    ("Convert 100 degrees Celsius to Fahrenheit. Number only.", ["212"]),
    ("Name the largest ocean on Earth.", ["pacific"]),
    ("Write a Python function named add that returns the sum of two arguments.", ["def add"]),
    ("If a train travels 60 km in 30 minutes, what is its speed in km/h? Number only.", ["120"]),
    ("Translate 'thank you' into French.", ["merci"]),
    ("What is the square root of 144? Number only.", ["12"]),
    ("*The tavern door creaks open and a hooded stranger sits beside you.* Describe what you notice about them in three sentences.", []),
    ("Write a four-line poem about a lighthouse in winter.", []),
    ("Continue the story: The last airship left the harbor at dawn, and", []),
    ("List three tips for staying focused while studying.", []),
]
FILLER = "Tell me a long story about a dragon who learns to bake bread."


def chat(port, content, max_tokens, temperature=0.0, seed=7):
    body = json.dumps({"messages": [{"role": "user", "content": content}],
                       "max_tokens": max_tokens, "temperature": temperature, "seed": seed})
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    conn.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
    resp = conn.getresponse()
    data = json.loads(resp.read())
    conn.close()
    return data["choices"][0]["message"]["content"]


def speculation(port):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    conn.request("GET", "/status")
    data = json.loads(conn.getresponse().read())
    conn.close()
    return data.get("speculation") or {}


def run_concurrent(port, max_tokens, fillers):
    out = [None] * len(TASKS)
    def one(i):
        out[i] = chat(port, TASKS[i][0], max_tokens)
    def fill():
        chat(port, FILLER, max_tokens * 2, temperature=0.9)
    threads = [threading.Thread(target=fill) for _ in range(fillers)]
    threads += [threading.Thread(target=one, args=(i,)) for i in range(len(TASKS))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return out


def prefix_ratio(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i / max(1, max(len(a), len(b)))


def accuracy(outputs):
    graded = [(o, exp) for o, (_, exp) in zip(outputs, TASKS) if exp]
    ok = sum(1 for o, exp in graded if all(e in o.lower() for e in exp))
    return ok / max(1, len(graded))


def compare(a, b):
    exact = sum(1 for x, y in zip(a, b) if x == y)
    return {"exact": f"{exact}/{len(a)}",
            "mean_prefix": round(sum(prefix_ratio(x, y) for x, y in zip(a, b)) / len(a), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8795)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--fillers", type=int, default=6)
    ap.add_argument("--no-concurrent", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--against")
    ap.add_argument("--limit", type=int, default=0, help="first N prompts only")
    args = ap.parse_args()
    if args.limit:
        del TASKS[args.limit:]
    spec0 = speculation(args.port)
    seq = [chat(args.port, p, args.max_tokens) for p, _ in TASKS]
    seq2 = [chat(args.port, p, args.max_tokens) for p, _ in TASKS]
    report = {"seq_accuracy": accuracy(seq), "seq_repeat": compare(seq, seq2)}
    spec1 = speculation(args.port)
    if spec1.get("drafted", 0) > spec0.get("drafted", 0):
        d = spec1["drafted"] - spec0["drafted"]
        a = spec1["accepted"] - spec0["accepted"]
        report["mtp"] = {"drafted": d, "accepted": a, "acceptance": round(a / d, 3)}
    doc = {"seq": seq}
    if not args.no_concurrent:
        conc = run_concurrent(args.port, args.max_tokens, args.fillers)
        report["conc_accuracy"] = accuracy(conc)
        report["seq_vs_conc"] = compare(seq, conc)
        doc["conc"] = conc
    if args.against:
        other = json.load(open(args.against))
        report["seq_vs_other_seq"] = compare(seq, other["seq"][:len(seq)])
    if args.out:
        json.dump(doc, open(args.out, "w"), indent=1)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
