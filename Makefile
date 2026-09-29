CC ?= cc
override CFLAGS += -std=c17 -Wall -Wextra -Wpedantic -Werror -O2
override CPPFLAGS += -D_POSIX_C_SOURCE=200809L -Isrc -Iinclude

BIN := build/omniserve
CORE_SRC := src/ocapacity.c src/ohttp.c src/oimage.c src/ojson.c src/olog.c \
	src/omatte.c src/onsfw.c src/oproxy.c src/osched.c src/ohost.c src/ovram.c \
	src/ospec.c src/oscale.c src/otext.c src/otune.c
BIN_SRC := src/main.c src/docs.c src/backend_llama.c src/backend_sd.c $(CORE_SRC)
TESTS := build/test_admission build/test_breaker build/test_wavwrap
JOBCTL := build/omni-job
JOB_TEST := build/test_jobs
MUSIC3C := build/music3c
MUSIC3C_DEPLOY := build/music3c-deploy
MUSIC3C_TEST := build/test_music3c
WAVWRAP := build/wavwrap
DEPLOY_CC ?= $(firstword $(wildcard /usr/bin/gcc /usr/bin/cc) cc)
SQLITE_LIB := $(shell pkg-config --variable=libdir sqlite3)/libsqlite3.so
ifneq ($(wildcard $(CONDA_PREFIX)/lib/libsqlite3.so),)
SQLITE_LIB := $(CONDA_PREFIX)/lib/libsqlite3.so
override LDFLAGS += -Wl,-rpath,$(CONDA_PREFIX)/lib
endif

.PHONY: all run test clean

all: $(BIN) $(JOBCTL) $(MUSIC3C) $(WAVWRAP)

build:
	mkdir -p $@

$(BIN): build $(BIN_SRC)
	$(CC) $(CPPFLAGS) $(CFLAGS) -Wno-overlength-strings $(BIN_SRC) -pthread -ldl -lm -o $@

build/test_admission: build tests/test_admission.c src/admission.c src/admission.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -UNDEBUG tests/test_admission.c src/admission.c -o $@

$(JOBCTL): build src/jobctl.c src/jobs.c src/jobs.h
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) src/jobctl.c src/jobs.c $(SQLITE_LIB) -o $@

$(JOB_TEST): build tests/test_jobs.c src/jobs.c src/jobs.h
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -UNDEBUG tests/test_jobs.c src/jobs.c $(SQLITE_LIB) -o $@

build/test_breaker: build tests/test_breaker.c src/breaker.c src/breaker.h
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_breaker.c src/breaker.c -o $@

$(WAVWRAP): build src/wavwrap_main.c src/wavwrap.c src/wavwrap.h
	$(CC) $(CPPFLAGS) $(CFLAGS) src/wavwrap_main.c src/wavwrap.c -o $@

build/test_wavwrap: build tests/test_wavwrap.c src/wavwrap.c src/wavwrap.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -UNDEBUG tests/test_wavwrap.c src/wavwrap.c -o $@

build/bench_wavwrap: build tests/bench_wavwrap.c src/wavwrap.c src/wavwrap.h
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/bench_wavwrap.c src/wavwrap.c -o $@

NVCC ?= $(firstword $(wildcard /usr/local/cuda-12/bin/nvcc /usr/local/cuda/bin/nvcc) nvcc)
ifneq ($(shell command -v $(NVCC) 2>/dev/null),)
build/bench_wavwrap_cuda: build tests/bench_wavwrap_cuda.cu
	$(NVCC) -O2 -std=c++17 tests/bench_wavwrap_cuda.cu -o $@
endif

$(MUSIC3C): build music3c/main.c music3c/music3.c music3c/music3.h
	$(CC) $(CPPFLAGS) $(CFLAGS) music3c/main.c music3c/music3.c -lm -lpthread -o $@

# The deployed worker runs inside the SGLang-Omni image, so it is linked
# statically with the system toolchain: a build against another libc segfaults
# on the container's loader and the worker then never polls for jobs.
$(MUSIC3C_DEPLOY): build music3c/main.c music3c/music3.c music3c/music3.h
	$(DEPLOY_CC) -std=c17 -O2 -Wall -Wextra -D_POSIX_C_SOURCE=200809L -static -s \
		music3c/main.c music3c/music3.c -lm -lpthread -o $@
	$@ --selftest

$(MUSIC3C_TEST): build tests/test_music3c.c music3c/music3.c music3c/music3.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -Imusic3c -UNDEBUG tests/test_music3c.c music3c/music3.c -lm -o $@

run: $(BIN)
	$(BIN) --models models/models.csv

test: $(TESTS) $(JOB_TEST) $(MUSIC3C_TEST) $(MUSIC3C)
	build/test_admission
	build/test_wavwrap
	$(JOB_TEST)
	$(MUSIC3C_TEST)
	python3 -m pytest -q tests/test_runtime.py tests/test_person_detection.py tests/test_video_matting.py tests/test_music3_handler.py tests/test_music3_result_cache.py tests/test_wan_animate_2.py

ASAN_FLAGS := -fsanitize=address,undefined -fno-omit-frame-pointer
ASAN_CC := $(firstword $(wildcard /usr/bin/gcc /usr/bin/cc) gcc)
.PHONY: test-asan
test-asan: build
	$(ASAN_CC) $(CPPFLAGS) $(CFLAGS) $(ASAN_FLAGS) -UNDEBUG tests/test_admission.c src/admission.c -o build/test_admission_asan
	$(ASAN_CC) $(CPPFLAGS) $(CFLAGS) $(ASAN_FLAGS) tests/test_breaker.c src/breaker.c -o build/test_breaker_asan
	$(ASAN_CC) $(CPPFLAGS) $(CFLAGS) $(ASAN_FLAGS) -UNDEBUG tests/test_jobs.c src/jobs.c $(LDFLAGS) $(SQLITE_LIB) -o build/test_jobs_asan
	$(ASAN_CC) $(CPPFLAGS) $(CFLAGS) $(ASAN_FLAGS) -Imusic3c -UNDEBUG tests/test_music3c.c music3c/music3.c -lm -o build/test_music3c_asan
	build/test_admission_asan
	build/test_breaker_asan
	build/test_jobs_asan
	build/test_music3c_asan

clean:
	rm -rf build
