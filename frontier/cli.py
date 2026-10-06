from __future__ import annotations

import argparse
import json
import os
import sys
import time

from .ledger import DEFAULT_DB, Ledger
from .policy import build_routing, load_seeds, write_routing
from .router import DEFAULT_ROUTING

DEFAULT_LOGS = "/nvme0n1-disk/data/omniserve-frontier/gateway-ra2.jsonl"


def api_key(args) -> str:
    if args.key_file:
        for line in open(args.key_file):
            if line.startswith("RUNPOD_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"')
    return os.getenv("RUNPOD_API_KEY", "")


def ingest(ledger: Ledger, logs: str) -> dict:
    return {path: ledger.ingest_jsonl(path) for path in filter(None, logs.split(","))}


def build(ledger: Ledger, args) -> dict:
    routing = build_routing(load_seeds(args.seeds), ledger, days=args.days)
    write_routing(routing, args.out)
    return routing


def table(routing: dict) -> str:
    lines = ["workload   candidate                    p50_s   p95_s   $/job     quality frontier n"]
    for name, spec in routing["workloads"].items():
        for c in spec["candidates"]:
            p50 = c.get("p50_ms")
            p95 = c.get("p95_ms")
            usd = c.get("usd_per_job")
            lines.append(f"{name:<10} {c['id']:<28} {p50 / 1000 if p50 else float('nan'):>6.1f} "
                         f"{p95 / 1000 if p95 else float('nan'):>7.1f} {usd if usd is not None else float('nan'):>8.5f} "
                         f"{'ok' if c.get('quality_ok') else 'no':<7} {'*' if c.get('frontier') else ' ':<8} "
                         f"{c.get('n', c.get('source'))}{'' if c.get('available', True) else ' unavailable'}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m frontier")
    parser.add_argument("command", choices=("ingest", "reconcile", "build", "summary", "metrics", "table", "cycle"))
    parser.add_argument("--db", default=os.getenv("FRONTIER_LEDGER_DB", DEFAULT_DB))
    parser.add_argument("--out", default=os.getenv("FRONTIER_ROUTING_FILE", DEFAULT_ROUTING))
    parser.add_argument("--seeds", default=os.getenv("FRONTIER_SEEDS"))
    parser.add_argument("--logs", default=os.getenv("FRONTIER_GATEWAY_LOGS", DEFAULT_LOGS))
    parser.add_argument("--days", type=float, default=7)
    parser.add_argument("--hours", type=float, default=24)
    parser.add_argument("--key-file", default=os.getenv("FRONTIER_RUNPOD_KEY_FILE"))
    parser.add_argument("--reconcile-every-h", type=float, default=20)
    args = parser.parse_args(argv)
    ledger = Ledger(args.db)
    if args.command == "ingest":
        print(json.dumps(ingest(ledger, args.logs)))
    elif args.command == "reconcile":
        print(json.dumps(ledger.reconcile(api_key(args), days=int(args.days)), indent=1))
    elif args.command == "build":
        print(table(build(ledger, args)))
    elif args.command == "table":
        print(table(build_routing(load_seeds(args.seeds), ledger, days=args.days)))
    elif args.command == "summary":
        print(json.dumps(ledger.summary(args.hours), indent=1))
    elif args.command == "metrics":
        sys.stdout.write(ledger.prometheus(args.hours))
    elif args.command == "cycle":
        result = {"ingested": ingest(ledger, args.logs)}
        last = float(ledger.meta("last_reconcile", "0") or 0)
        key = api_key(args)
        if key and time.time() - last > args.reconcile_every_h * 3600:
            try:
                result["reconciled"] = len(ledger.reconcile(key, days=int(args.days)))
            except OSError as exc:
                result["reconcile_error"] = str(exc)
        routing = build(ledger, args)
        result["frontier"] = {k: v["frontier"] for k, v in routing["workloads"].items()}
        print(json.dumps(result))
    return 0
