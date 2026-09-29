#ifndef OMNISERVE_BREAKER_H
#define OMNISERVE_BREAKER_H

#include <stdbool.h>
#include <time.h>

typedef struct {
    unsigned failures;
    unsigned failure_limit;
    time_t open_until;
    unsigned cooldown_seconds;
} CircuitBreaker;

void breaker_init(CircuitBreaker *breaker, unsigned failure_limit,
                  unsigned cooldown_seconds);
bool breaker_allows(const CircuitBreaker *breaker, time_t now);
void breaker_record_success(CircuitBreaker *breaker);
void breaker_record_failure(CircuitBreaker *breaker, time_t now);
const char *breaker_state(const CircuitBreaker *breaker, time_t now);

#endif
