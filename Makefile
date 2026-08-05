CC ?= cc
CFLAGS ?= -std=c17 -Wall -Wextra -Wpedantic -Werror -O2
CPPFLAGS ?= -D_POSIX_C_SOURCE=200809L -Isrc

BIN := build/omniserve
TEST := build/test_admission

.PHONY: all run test clean

all: $(BIN)

build:
	mkdir -p $@

$(BIN): build src/main.c src/admission.c src/admission.h
	$(CC) $(CPPFLAGS) $(CFLAGS) src/main.c src/admission.c -o $@

$(TEST): build tests/test_admission.c src/admission.c src/admission.h
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_admission.c src/admission.c -o $@

run: $(BIN)
	$(BIN) --models models/models.csv

test: $(TEST)
	$(TEST)

clean:
	rm -rf build
