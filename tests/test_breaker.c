#include "breaker.h"

#include <assert.h>
#include <string.h>

int main(void) {
    CircuitBreaker breaker;
    breaker_init(&breaker, 3, 30);
    assert(breaker_allows(&breaker, 100));
    breaker_record_failure(&breaker, 100);
    breaker_record_failure(&breaker, 101);
    assert(strcmp(breaker_state(&breaker, 101), "closed-degraded") == 0);
    breaker_record_failure(&breaker, 102);
    assert(!breaker_allows(&breaker, 120));
    assert(strcmp(breaker_state(&breaker, 120), "open") == 0);
    assert(breaker_allows(&breaker, 132));
    breaker_record_success(&breaker);
    assert(strcmp(breaker_state(&breaker, 132), "closed") == 0);
    return 0;
}
