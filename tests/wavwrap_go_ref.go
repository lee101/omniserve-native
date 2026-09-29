package main

import (
	"io"
	"os"
)

// Reference implementation matching ringnz backend-go wavFromPCM16
// (single allocation, little-endian stores) used by bench_wavwrap.sh.
func main() {
	pcm, err := io.ReadAll(os.Stdin)
	if err != nil {
		panic(err)
	}
	rate := uint32(24000)
	if v := os.Getenv("WAVWRAP_RATE"); v == "16000" {
		rate = 16000
	}
	out := make([]byte, 44+len(pcm))
	copy(out[0:4], "RIFF")
	lePutU32(out[4:8], uint32(36+len(pcm)))
	copy(out[8:12], "WAVE")
	copy(out[12:16], "fmt ")
	lePutU32(out[16:20], 16)
	lePutU16(out[20:22], 1)
	lePutU16(out[22:24], 1)
	lePutU32(out[24:28], rate)
	lePutU32(out[28:32], rate*2)
	lePutU16(out[32:34], 2)
	lePutU16(out[34:36], 16)
	copy(out[36:40], "data")
	lePutU32(out[40:44], uint32(len(pcm)))
	copy(out[44:], pcm)
	if _, err := os.Stdout.Write(out); err != nil {
		panic(err)
	}
}

func lePutU16(b []byte, v uint16) {
	b[0] = byte(v)
	b[1] = byte(v >> 8)
}

func lePutU32(b []byte, v uint32) {
	b[0] = byte(v)
	b[1] = byte(v >> 8)
	b[2] = byte(v >> 16)
	b[3] = byte(v >> 24)
}
