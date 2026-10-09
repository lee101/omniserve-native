#include "olora_route.h"

#include "ojson.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

bool olora_route_enabled(void) {
    const char *value = getenv("OMNISERVE_NATIVE_LORA_AUTO_ROUTE");
    return value && (strcmp(value, "1") == 0 || strcmp(value, "true") == 0 ||
                     strcmp(value, "yes") == 0 || strcmp(value, "on") == 0);
}

static char *lower_copy(const char *text) {
    size_t len = strlen(text);
    char *out = malloc(len + 1);
    if (!out) return NULL;
    for (size_t i = 0; i < len; ++i) out[i] = (char)tolower((unsigned char)text[i]);
    out[len] = 0;
    return out;
}

static bool word_char(char c) {
    return isalnum((unsigned char)c) || (unsigned char)c >= 0x80;
}

static bool phrase_match(const char *haystack, const char *phrase) {
    char needle[96];
    size_t len = strlen(phrase);
    if (!len || len >= sizeof needle) return false;
    bool prefix = phrase[len - 1] == '*';
    if (prefix && --len == 0) return false;
    memcpy(needle, phrase, len);
    needle[len] = 0;
    for (const char *at = haystack; (at = strstr(at, needle)) != NULL; ++at) {
        bool left = at == haystack || !word_char(at[-1]);
        bool right = prefix || !word_char(at[len]);
        if (left && right) return true;
    }
    return false;
}

static bool safe_id(const char *value) {
    if (!value[0] || strlen(value) > 128) return false;
    for (const unsigned char *p = (const unsigned char *)value; *p; ++p) {
        if (!(isalnum(*p) || *p == '-' || *p == '_' || *p == '.')) return false;
    }
    return true;
}

static bool any_keyword(const char *js, const oj_tok *toks, int ntoks, int arr,
                        const char *prompt) {
    for (int i = 0; i < toks[arr].size; ++i) {
        int item = oj_arr_at(toks, ntoks, arr, i);
        if (item < 0 || toks[item].type != OJ_STRING) continue;
        char raw[96];
        if (!oj_unescape(js, &toks[item], raw, sizeof raw)) continue;
        char *keyword = lower_copy(raw);
        bool hit = keyword && phrase_match(prompt, keyword);
        free(keyword);
        if (hit) return true;
    }
    return false;
}

static bool fill_pick(const char *js, const oj_tok *toks, int ntoks, int route,
                      olora_route_pick *pick) {
    memset(pick, 0, sizeof *pick);
    int name = oj_obj_get(js, toks, ntoks, route, "name");
    if (name >= 0 && toks[name].type == OJ_STRING) {
        oj_unescape(js, &toks[name], pick->name, sizeof pick->name);
    }
    int prefix = oj_obj_get(js, toks, ntoks, route, "prefix");
    if (prefix >= 0 && toks[prefix].type == OJ_STRING) {
        oj_unescape(js, &toks[prefix], pick->prefix, sizeof pick->prefix);
    }
    int loras = oj_obj_get(js, toks, ntoks, route, "loras");
    if (loras < 0 || toks[loras].type != OJ_ARRAY) return false;
    for (int i = 0; i < toks[loras].size && pick->lora_count < OLORA_ROUTE_MAX_LORAS; ++i) {
        int item = oj_arr_at(toks, ntoks, loras, i);
        if (item < 0 || toks[item].type != OJ_OBJECT) return false;
        int id = oj_obj_get(js, toks, ntoks, item, "id");
        if (id < 0 || toks[id].type != OJ_STRING) return false;
        char *slot = pick->ids[pick->lora_count];
        if (!oj_unescape(js, &toks[id], slot, sizeof pick->ids[0]) || !safe_id(slot)) return false;
        double scale = 1.0;
        int scale_token = oj_obj_get(js, toks, ntoks, item, "scale");
        if (scale_token >= 0) scale = oj_number(js, &toks[scale_token], NAN);
        if (!isfinite(scale) || scale < -4.0 || scale > 4.0) return false;
        pick->scales[pick->lora_count++] = (float)scale;
    }
    return pick->lora_count > 0;
}

bool olora_route_select_json(const char *json, size_t json_len, const char *prompt,
                             olora_route_pick *pick) {
    if (!json || !prompt || !pick) return false;
    int cap = json_len + 2 < 16384 ? (int)json_len + 2 : 16384;
    oj_tok *toks = calloc((size_t)cap, sizeof *toks);
    if (!toks) return false;
    int ntoks = oj_parse(json, json_len, toks, cap);
    int routes = ntoks > 0 && toks[0].type == OJ_OBJECT
        ? oj_obj_get(json, toks, ntoks, 0, "routes") : -1;
    char *lowered = lower_copy(prompt);
    bool found = false;
    if (routes >= 0 && toks[routes].type == OJ_ARRAY && lowered) {
        for (int r = 0; r < toks[routes].size && !found; ++r) {
            int route = oj_arr_at(toks, ntoks, routes, r);
            if (route < 0 || toks[route].type != OJ_OBJECT) continue;
            int skip = oj_obj_get(json, toks, ntoks, route, "exclude");
            if (skip >= 0 && toks[skip].type == OJ_ARRAY &&
                any_keyword(json, toks, ntoks, skip, lowered)) continue;
            int keywords = oj_obj_get(json, toks, ntoks, route, "keywords");
            bool match = keywords < 0 ||
                (toks[keywords].type == OJ_ARRAY &&
                 any_keyword(json, toks, ntoks, keywords, lowered));
            if (match) found = fill_pick(json, toks, ntoks, route, pick);
        }
    }
    free(lowered);
    free(toks);
    if (!found) memset(pick, 0, sizeof *pick);
    return found;
}

bool olora_route_select(const char *prompt, olora_route_pick *pick) {
    const char *path = getenv("OMNISERVE_NATIVE_LORA_ROUTES");
    if (!path || !path[0]) return false;
    FILE *file = fopen(path, "rb");
    if (!file) return false;
    char *data = malloc(256 * 1024 + 1);
    size_t len = data ? fread(data, 1, 256 * 1024 + 1, file) : 0;
    fclose(file);
    if (!data || len > 256 * 1024) {
        free(data);
        return false;
    }
    data[len] = 0;
    bool ok = olora_route_select_json(data, len, prompt, pick);
    free(data);
    return ok;
}
