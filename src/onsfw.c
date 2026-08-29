#include "onsfw.h"

#include <ctype.h>
#include <stddef.h>
#include <string.h>

static const char *const nsfw_words[] = {
    "nsfw", "nude", "nudity", "naked",
    "porn", "porno", "pornography", "pornographic",
    "explicit", "sexual", "sexually", "sex", "intercourse",
    "erotic", "erotica", "fetish", "hentai", "xxx",
    "topless", "orgasm", "masturbation", "masturbate",
    "blowjob", "handjob", "dildo", "genital", "genitals",
    "vagina", "penis", "breast", "breasts", "nipple", "nipples",
    "areola", "anal", "anus", "semen", "cum", "ejaculate",
    "ejaculation",
};

static bool is_word_char(unsigned char c) {
    return isalnum(c) != 0 || c == '_';
}

static bool is_nsfw_word(const char *start, size_t len) {
    for (size_t i = 0; i < sizeof nsfw_words / sizeof nsfw_words[0]; ++i) {
        if (strlen(nsfw_words[i]) != len) continue;
        bool equal = true;
        for (size_t j = 0; j < len; ++j) {
            if ((char)tolower((unsigned char)start[j]) != nsfw_words[i][j]) {
                equal = false;
                break;
            }
        }
        if (equal) return true;
    }
    return false;
}

bool onsfw_prompt_has_word(const char *prompt) {
    if (!prompt) return false;
    const unsigned char *p = (const unsigned char *)prompt;
    while (*p) {
        while (*p && !is_word_char(*p)) ++p;
        const unsigned char *start = p;
        while (*p && is_word_char(*p)) ++p;
        if (p > start && is_nsfw_word((const char *)start, (size_t)(p - start))) {
            return true;
        }
    }
    return false;
}
