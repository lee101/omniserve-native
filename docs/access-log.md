# Access log

`/metrics` and `/errors` answer "is the gateway failing, on which route". They
cannot answer "who called, carrying which trust markers, how long did it take" —
the question that actually comes up after a suspected bypass, an abuse report, or
a bill that does not match the traffic. The access log answers that one, and is
deliberately built so that turning it on cannot cost throughput and cannot write
a credential that was supplied as a header value, a cookie or a query parameter.

It is not a redactor for credentials a caller puts in the *path*: the request
target is recorded (escaped and truncated), so a design that puts a token in a
path segment would be logged. No route on this gateway does that today, and that
is an assumption to re-check rather than a guarantee the log enforces.

## Line format

One line per request, written by a dedicated drain thread:

```
<ISO-8601 UTC ms> peer=<addr> method=<M> path="<escaped>" model=<name|-> status=<code> dur_ms=<f> internal=<0|1> hdrs=<names|->
```

Real samples:

```
2026-07-28T20:29:17.916Z peer=127.0.0.1 method=GET model=- status=200 dur_ms=0.02 internal=1 hdrs=- path="/v1/models"
2026-07-28T20:31:02.418Z peer=127.0.0.1 method=POST model=- status=401 dur_ms=1.87 internal=0 hdrs=X-Forwarded-For,Authorization,X-API-Key path="/v1/chat/completions"
2026-07-28T20:29:55.035Z event=access_log_drops dropped=599621 emitted=379 unwritten=0
```

- `internal` is the gateway's own `request_is_internal()` verdict (loopback peer
  **and** no relay header), recorded so a bypass can be reconstructed later
  rather than re-argued.
- `hdrs` is the list of trust-relevant headers that were **present**. The list is
  the router's relayed-header array (`relayed_headers` in `src/main.c`, the same
  one the bypass check walks, registered rather than copied) plus a second,
  independent list in `src/olog.c`: `Authorization`, `X-API-Key`,
  `X-Rapid-API-Key`, `secret`, `Cookie`, `X-Omniserve-Internal` and
  `X-Omniserve-Tier`. Registering the router's array rather than copying it means
  a header added to the trust check is logged the day it is added.
- `model` is the locally served model when a local handler runs. The inbound
  `model` field of the request body is **never** parsed: no handler parses it
  today, and adding a JSON parse to the hot path purely for logging is not worth
  it, so proxied requests log `-`.

## What is never written

Header **values**, the request body, prompt text, and the query string — the
query can carry `?secret=`/`?token=`, so it is dropped entirely rather than
filtered. Only header *names* appear, and only from the fixed list above.

The path is truncated to 192 escaped bytes (`+` marks truncation) and every byte
outside printable ASCII, plus `"` and `\`, is escaped as `\xHH`. A request for
`/inject"<CR>marker<0x01>` therefore produces exactly one line, not two: log
injection through the request target is the one input a caller fully controls.

## Off the critical path

Workers format into a fixed stack buffer (no allocation), copy the line into a
bounded lock-free ring (4096 slots x 512 B = 2 MiB), and return. A worker that
finds the ring full **drops the line and increments a counter** — it never waits
on disk. Drops are visible two ways: `omniserve_access_log_dropped_total` in
`/metrics`, and an `event=access_log_drops` line the drain thread writes into the
log itself at most every 10s while the count is changing.

The drain thread batches up to 32 KiB per `write(2)`.

## Retention: 128 MiB, hard (~1 month)

Rotation is in-process, not logrotate — a cap enforced by a tool that may not be
installed on a given box is not a cap. At 32 MiB the file is renamed
`access.log -> .1 -> .2 -> .3` and the oldest generation is overwritten.

If the rename cannot happen at all — read-only directory, ENOSPC, the name held
open elsewhere — the drain thread truncates the current file instead of
appending past the cap, and refuses the write if even that fails. Losing the
oldest lines is the lesser failure against filling a disk that other services
share. Lines drained from the ring but not written are counted as `unwritten=`
in the periodic drop line, so a silently unwritable log is visible rather than
indistinguishable from no traffic.

The cap is per process. Two gateways pointed at the same directory would each
enforce it independently against the same files; the systemd unit runs one.

**4 files x 32 MiB = 128 MiB maximum on disk**, forever. Still small against the
~1 GB of generated data this host already carries.

The size is set by the question the log exists to answer. Measured in production:
~185 B/line at ~25k requests/day, so 128 MiB is roughly **a month** of history,
about 700k requests. A 32 MiB cap would have expired at ~7 days — the same age as
Cloudflare's edge retention, and therefore the same blind spot that made the
bypass audit inconclusive. Retention below the edge window would have been
retention that answers nothing.

## Configuration

| Variable | Default | Effect |
| --- | --- | --- |
| `OMNISERVE_ACCESS_LOG` | on | `0`/`false`/`no` disables logging entirely |
| `OMNISERVE_ACCESS_LOG_DIR` | `/var/log/omniserve` | Directory for `access.log`; created if missing, falls back to `/tmp/omniserve-access.log` with a stderr note if unusable |
| `OMNISERVE_ACCESS_LOG_MAX_BYTES` | 33554432 | May only *lower* the per-file cap; the 128 MiB ceiling is not negotiable from the environment |

## Measured overhead

`onative_bench` against `/v1/models` (cheapest real route, no inference),
300k requests over 16 keep-alive connections, gateway restarted between runs and
runs interleaved on/off five times:

| | throughput (best of 5) | throughput (median of 5) | p50 latency (best run) |
| --- | --- | --- | --- |
| logging off | 136,437 req/s | 127,190 req/s | 111.5 us |
| logging on | 132,128 req/s | 125,514 req/s | 107.5 us |

That is a ~1-3% throughput difference and no measurable latency change — below
the noise floor of this box, which was running at load average ~55 (72 cores)
from other tenants during the measurement, so treat 3% as an upper bound rather
than a point estimate. Drops were zero at 119k req/s across 128 connections;
forcing them (ring shrunk to 8 slots) cost throughput nothing and produced the
`access_log_drops` line as designed.
