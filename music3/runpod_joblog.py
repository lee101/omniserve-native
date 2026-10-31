"""Drop-in per-job log shipping for RunPod serverless handlers (stdlib only; copy this file next to your handler).

    import runpod, runpod_joblog
    runpod.serverless.start({"handler": runpod_joblog.wrap(handler, "netwrck/music-generator")})

Tees stdout/stderr, and when each job ends (success, exception, or timeout that raises) PUTs the job's gzipped log slice to
s3://<bucket>/<site>/<service>/YYYY/MM/DD/<job>.log.gz in the shared PRIVATE R2 bucket. Never fails or delays a job (20 s cap).
Credentials come from the endpoint env: LOG_R2_ACCOUNT_ID, LOG_R2_ACCESS_KEY_ID, LOG_R2_SECRET_ACCESS_KEY (+ optional LOG_R2_BUCKET,
default "private"). A job input `_log_upload_url` (presigned PUT, as manifoldgen-site sends) is used when no credentials are set.
Logs include job inputs, so keep the bucket private. Rolled up fortnightly by manifoldgen-site/scripts/h3_log_rollup.py --prefix.
"""
import datetime as dt, functools, gzip, hashlib, hmac, http.client, json, os, sys, threading, time, traceback, urllib.parse
from collections import deque

_lines, _seq, _lock = deque(maxlen=20000), 0, threading.Lock()


class _Tee:
    def __init__(self, stream): self._s, self._p = stream, ""
    def write(self, text):
        global _seq
        n = self._s.write(text)
        *done, self._p = (self._p + text).split("\n")
        if done:
            stamp = time.strftime("%H:%M:%S")
            with _lock:
                for line in done:
                    _seq += 1; _lines.append((_seq, f"{stamp} {line[:2000]}"))
        return n
    def __getattr__(self, name): return getattr(self._s, name)


def _hmac(key, msg): return hmac.new(key, msg.encode(), hashlib.sha256).digest()


def _signed_put(host, bucket, key, body, account_key, secret):
    now = dt.datetime.now(dt.timezone.utc); amz, day = now.strftime("%Y%m%dT%H%M%SZ"), now.strftime("%Y%m%d")
    path = "/" + bucket + "/" + urllib.parse.quote(key, safe="/")
    payload = hashlib.sha256(body).hexdigest()
    headers = {"host": host, "x-amz-content-sha256": payload, "x-amz-date": amz}
    signed = ";".join(sorted(headers))
    canon = "\n".join(["PUT", path, "", *(f"{k}:{headers[k]}" for k in sorted(headers)), "", signed, payload])
    scope = f"{day}/auto/s3/aws4_request"
    to_sign = "\n".join(["AWS4-HMAC-SHA256", amz, scope, hashlib.sha256(canon.encode()).hexdigest()])
    k = _hmac(_hmac(_hmac(_hmac(("AWS4" + secret).encode(), day), "auto"), "s3"), "aws4_request")
    sig = hmac.new(k, to_sign.encode(), hashlib.sha256).hexdigest()
    headers["authorization"] = f"AWS4-HMAC-SHA256 Credential={account_key}/{scope}, SignedHeaders={signed}, Signature={sig}"
    conn = http.client.HTTPSConnection(host, timeout=20)
    try:
        conn.request("PUT", path, body=body, headers=headers); return conn.getresponse().status < 300
    finally: conn.close()


def ship(prefix, since, job, error, started, upload_url=None):
    try:
        acct, ak, sk = (os.environ.get(f"LOG_R2_{n}", "") for n in ("ACCOUNT_ID", "ACCESS_KEY_ID", "SECRET_ACCESS_KEY"))
        if not ((acct and ak and sk) or upload_url): return False
        with _lock: tail = [l for s, l in _lines if s > since]
        inp = job.get("input") if isinstance(job, dict) else None
        head = {"job_id": (job or {}).get("id"), "status": "failed" if error else "ok", "seconds": round(time.monotonic() - started, 2),
                "error": "".join(traceback.format_exception_only(type(error), error)).strip() if error else None,
                "input": {k: (v if len(json.dumps(v, default=str)) < 4000 else "<large>") for k, v in (inp or {}).items() if not str(k).startswith("_")}}
        body = gzip.compress(("\n".join([json.dumps(head, default=str), *tail]) + "\n").encode(), 6)
        if acct and ak and sk:
            key = f"{prefix.strip('/')}/{dt.datetime.now(dt.timezone.utc):%Y/%m/%d}/{head['job_id'] or int(time.time()*1000)}.log.gz"
            return _signed_put(f"{acct}.r2.cloudflarestorage.com", os.environ.get("LOG_R2_BUCKET", "private"), key, body, ak, sk)
        u = urllib.parse.urlparse(upload_url)
        if u.scheme != "https" or not (u.hostname or "").endswith(".r2.cloudflarestorage.com"): return False
        c = http.client.HTTPSConnection(u.hostname, timeout=20)
        try: c.request("PUT", u.path + "?" + u.query, body=body, headers={"Content-Type": "application/gzip"}); return c.getresponse().status < 300
        finally: c.close()
    except Exception as exc:
        print(f"joblog upload failed: {exc}", flush=True); return False


def wrap(handler, service):
    """service like "netwrck/music-generator" -> logs under <service>/YYYY/MM/DD/ in the private bucket."""
    if not isinstance(sys.stdout, _Tee): sys.stdout, sys.stderr = _Tee(sys.stdout), _Tee(sys.stderr)

    @functools.wraps(handler)
    def run(job, *a, **kw):
        with _lock: since = _seq
        started, error = time.monotonic(), None
        try: return handler(job, *a, **kw)
        except BaseException as e: error = e; raise
        finally:
            ship(service + "/worker-logs", since, job, error, started, (job.get("input") or {}).get("_log_upload_url") if isinstance(job, dict) and isinstance(job.get("input"), dict) else None)
    return run
