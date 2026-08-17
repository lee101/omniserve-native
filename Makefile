CC ?= cc
override CFLAGS += -std=c17 -Wall -Wextra -Wpedantic -Werror -O2
override CPPFLAGS += -D_POSIX_C_SOURCE=200809L -Isrc

BIN := build/omniserve
TESTS := build/test_admission build/test_breaker
JOBCTL := build/omni-job
JOB_TEST := build/test_jobs
MUSIC3C := build/music3c
MUSIC3C_TEST := build/test_music3c
SQLITE_LIB := $(shell pkg-config --variable=libdir sqlite3)/libsqlite3.so
ifneq ($(wildcard $(CONDA_PREFIX)/lib/libsqlite3.so),)
SQLITE_LIB := $(CONDA_PREFIX)/lib/libsqlite3.so
override LDFLAGS += -Wl,-rpath,$(CONDA_PREFIX)/lib
endif

.PHONY: all run test clean

all: $(BIN) $(JOBCTL) $(MUSIC3C)

build:
	mkdir -p $@

$(BIN): build src/main.c src/admission.c src/admission.h src/breaker.c src/breaker.h
	$(CC) $(CPPFLAGS) $(CFLAGS) src/main.c src/admission.c src/breaker.c -o $@

build/test_admission: build tests/test_admission.c src/admission.c src/admission.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -UNDEBUG tests/test_admission.c src/admission.c -o $@

$(JOBCTL): build src/jobctl.c src/jobs.c src/jobs.h
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) src/jobctl.c src/jobs.c $(SQLITE_LIB) -o $@

$(JOB_TEST): build tests/test_jobs.c src/jobs.c src/jobs.h
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -UNDEBUG tests/test_jobs.c src/jobs.c $(SQLITE_LIB) -o $@

build/test_breaker: build tests/test_breaker.c src/breaker.c src/breaker.h
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_breaker.c src/breaker.c -o $@

$(MUSIC3C): build music3c/main.c music3c/music3.c music3c/music3.h
	$(CC) $(CPPFLAGS) $(CFLAGS) music3c/main.c music3c/music3.c -lm -o $@

$(MUSIC3C_TEST): build tests/test_music3c.c music3c/music3.c music3c/music3.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -Imusic3c -UNDEBUG tests/test_music3c.c music3c/music3.c -lm -o $@

run: $(BIN)
	$(BIN) --models models/models.csv

test: $(TESTS) $(JOB_TEST) $(MUSIC3C_TEST)
	build/test_admission
	build/test_breaker
	$(JOB_TEST)
	$(MUSIC3C_TEST)
	python3 -m pytest -q tests/test_runtime.py tests/test_person_detection.py tests/test_video_matting.py tests/test_music3_handler.py tests/test_wan_animate_2.py

clean:
	rm -rf build
