#include "breaker.h"

void breaker_init(CircuitBreaker *breaker, unsigned failure_limit,
                  unsigned cooldown_seconds) {
    breaker->failures = 0;
    breaker->failure_limit = failure_limit == 0 ? 3 : failure_limit;
    breaker->open_until = 0;
    breaker->cooldown_seconds = cooldown_seconds == 0 ? 30 : cooldown_seconds;
}

bool breaker_allows(const CircuitBreaker *breaker, time_t now) {
    return breaker->open_until == 0 || now >= breaker->open_until;
}

void breaker_record_success(CircuitBreaker *breaker) {
    breaker->failures = 0;
    breaker->open_until = 0;
}

void breaker_record_failure(CircuitBreaker *breaker, time_t now) {
    if (breaker->open_until != 0 && now < breaker->open_until) return;
    ++breaker->failures;
    if (breaker->failures >= breaker->failure_limit)
        breaker->open_until = now + (time_t)breaker->cooldown_seconds;
}

const char *breaker_state(const CircuitBreaker *breaker, time_t now) {
    if (breaker->open_until != 0 && now < breaker->open_until) return "open";
    if (breaker->failures != 0) return "closed-degraded";
    return "closed";
}
