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

static const char *const anime_words[] = {
    "anime", "manga", "hentai", "waifu", "husbando", "chibi", "ecchi",
    "otaku", "kawaii", "doujin", "doujinshi", "vtuber", "celshaded",
    "shonen", "shoujo", "seinen", "josei",
    "danbooru", "pixiv", "2d", "cartoon", "toon",
};

static bool is_word_char(unsigned char c) {
    return isalnum(c) != 0 || c == '_';
}

static bool word_in(const char *const *list, size_t n, const char *start, size_t len) {
    for (size_t i = 0; i < n; ++i) {
        if (strlen(list[i]) != len) continue;
        bool equal = true;
        for (size_t j = 0; j < len; ++j) {
            if ((char)tolower((unsigned char)start[j]) != list[i][j]) {
                equal = false;
                break;
            }
        }
        if (equal) return true;
    }
    return false;
}

static bool prompt_has(const char *prompt, const char *const *list, size_t n) {
    if (!prompt) return false;
    const unsigned char *p = (const unsigned char *)prompt;
    while (*p) {
        while (*p && !is_word_char(*p)) ++p;
        const unsigned char *start = p;
        while (*p && is_word_char(*p)) ++p;
        if (p > start && word_in(list, n, (const char *)start, (size_t)(p - start))) {
            return true;
        }
    }
    return false;
}

bool onsfw_prompt_has_word(const char *prompt) {
    return prompt_has(prompt, nsfw_words, sizeof nsfw_words / sizeof nsfw_words[0]);
}

bool onsfw_prompt_has_anime_word(const char *prompt) {
    return prompt_has(prompt, anime_words, sizeof anime_words / sizeof anime_words[0]);
}
