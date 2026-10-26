import http.client, json, sys, time, statistics
port = int(sys.argv[1])
def req(sysmsg, tail, n=1):
    body = json.dumps({"messages": [{"role": "system", "content": sysmsg}, {"role": "user", "content": tail}],
                       "max_tokens": n, "temperature": 0})
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    t = time.perf_counter()
    c.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
    r = json.loads(c.getresponse().read())
    return (time.perf_counter() - t) * 1000, r["usage"]["prompt_tokens"]
def doc(seed):
    return " ".join(f"Entry {seed}-{i}: the harbor ledger records crate {i * 7 % 101} bound for island {i % 13}." for i in range(130))
cold, warm = [], []
for s in range(4):
    d = doc(100 + s)
    ms, pt = req(d, "Summarize in one word.")
    cold.append(ms)
    for k in range(3):
        ms, _ = req(d, f"Question {k}: which island is crate 7 bound for?")
        warm.append(ms)
print(json.dumps({"prompt_tokens": pt, "cold_ms_med": round(statistics.median(cold), 1), "warm_ms_med": round(statistics.median(warm), 1)}))
