#include "otext.h"

static bool ascii_alnum(unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9');
}

static bool ascii_space(unsigned char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\f' || c == '\v';
}

bool otext_completion_needs_space(const char *prompt, size_t prompt_len,
                                  const char *continuation, size_t continuation_len,
                                  bool raw_completion) {
    if (!prompt || !prompt_len || !continuation || !continuation_len) return false;
    unsigned char last = (unsigned char)prompt[prompt_len - 1];
    unsigned char first = (unsigned char)continuation[0];
    if (ascii_space(last) || ascii_space(first)) return false;
    if (first == '.' || first == ',' || first == '!' || first == '?' ||
        first == ';' || first == ':' || first == ')' || first == '}' ||
        first == ']' || first == '\'' || first == '"') return false;
    if (last == '(' || last == '{' || last == '[' || last == '\'' || last == '"') return false;

    if (!raw_completion) return true;

    /* A lowercase fragment can be a real subword continuation ("n" + "ame").
     * An uppercase token immediately after an alphanumeric prompt is an
     * unmistakable missing word boundary for natural-language autocomplete. */
    return ascii_alnum(last) && first >= 'A' && first <= 'Z';
}
