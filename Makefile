CC ?= cc
CFLAGS ?= -std=c17 -Wall -Wextra -Wpedantic -Werror -O2
CPPFLAGS ?= -D_POSIX_C_SOURCE=200809L -Isrc

BIN := build/omniserve
TESTS := build/test_admission build/test_breaker

.PHONY: all run test clean

all: $(BIN)

build:
	mkdir -p $@

$(BIN): build src/main.c src/admission.c src/admission.h src/breaker.c src/breaker.h
	$(CC) $(CPPFLAGS) $(CFLAGS) src/main.c src/admission.c src/breaker.c -o $@

build/test_admission: build tests/test_admission.c src/admission.c src/admission.h
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_admission.c src/admission.c -o $@

build/test_breaker: build tests/test_breaker.c src/breaker.c src/breaker.h
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_breaker.c src/breaker.c -o $@

run: $(BIN)
	$(BIN) --models models/models.csv

test: $(TESTS)
	build/test_admission
	build/test_breaker

clean:
	rm -rf build
