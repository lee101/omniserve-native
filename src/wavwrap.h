#ifndef OMNISERVE_WAVWRAP_H
#define OMNISERVE_WAVWRAP_H

#include <stddef.h>
#include <stdint.h>

/* Wraps raw mono PCM16LE samples in a 44-byte RIFF/WAVE header.
 * One allocation, one copy: out = header || pcm.
 * Returns 0 and sets *out and *out_len (caller frees the buffer).
 * Returns -1 on invalid arguments, -2 on allocation failure. */
int wav_wrap_pcm16(const uint8_t *pcm, size_t pcm_len, uint32_t sample_rate,
                   uint8_t **out, size_t *out_len);

#endif
