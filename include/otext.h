#ifndef OTEXT_H
#define OTEXT_H

#include <stdbool.h>
#include <stddef.h>

/* Decide whether a continuation needs a separator before it is appended.
 * Raw completions preserve tokenizer whitespace and are repaired only for an
 * unmistakably lost uppercase boundary ("looking" + "I"). Chat completions
 * are separate turns, so ordinary adjacent words need a separator too. */
bool otext_completion_needs_space(const char *prompt, size_t prompt_len,
                                  const char *continuation, size_t continuation_len,
                                  bool raw_completion);

#endif
