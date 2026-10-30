#include "olora_route.h"

#include "cbed.h"
#include "ojson.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>

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

/* Operator-config LoRA files only: absolute, no traversal, readable. */
static bool safe_path(const char *value) {
    if (value[0] != '/' || strlen(value) > 256 || strstr(value, "..")) return false;
    return access(value, R_OK) == 0;
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

/* Semantic routing: a route may carry "positive" descriptions (what it is for) and "avoid"
 * descriptions (what it must not take). The prompt is embedded with the static int8 model (cbed);
 * a route scores max cos(prompt, positive) - avoid_weight * max cos(prompt, avoid) and the best
 * score above the route's "min_score" wins. Keyword routes remain as the fallback. */
static cbed_model *g_embed;
static pthread_once_t g_embed_once = PTHREAD_ONCE_INIT;

static void embed_load(void) {
    const char *dir = getenv("OMNISERVE_NATIVE_LORA_EMBED_DIR");
    if (dir && dir[0]) g_embed = cbed_load(dir);
    if (dir && dir[0] && !g_embed) fprintf(stderr, "lora-route: embed model not loadable from %s\n", dir);
}

static cbed_model *embed_model(void) {
    pthread_once(&g_embed_once, embed_load);
    return g_embed;
}

static float embed_cos(const cbed_model *m, const int8_t *q, float qn, const char *text) {
    int8_t v[512];
    float vn;
    if (cbed_dim(m) > (int)sizeof v) return -1.0f;
    cbed_embed_quantized(m, text, v, &vn, NULL);
    if (qn <= 0 || vn <= 0) return -1.0f;
    return (float)cbed_dot_i8(q, v, cbed_dim(m)) / (qn * vn);
}

static float embed_best(const char *js, const oj_tok *toks, int ntoks, int arr,
                        const cbed_model *m, const int8_t *q, float qn) {
    float best = -1.0f;
    for (int i = 0; i < toks[arr].size; ++i) {
        int item = oj_arr_at(toks, ntoks, arr, i);
        char raw[256];
        if (item < 0 || toks[item].type != OJ_STRING || !oj_unescape(js, &toks[item], raw, sizeof raw)) continue;
        float c = embed_cos(m, q, qn, raw);
        if (c > best) best = c;
    }
    return best;
}

static int embed_pick_route(const char *js, const oj_tok *toks, int ntoks, int routes,
                            const char *prompt, const char *lowered, const cbed_model *m) {
    int8_t q[512];
    float qn;
    if (cbed_dim(m) > (int)sizeof q) return -1;
    cbed_embed_quantized(m, prompt, q, &qn, NULL);
    double weight = 0.5;
    int top = oj_obj_get(js, toks, ntoks, 0, "avoid_weight");
    if (top >= 0) weight = oj_number(js, &toks[top], 0.5);
    int best = -1;
    float best_score = -2.0f;
    for (int r = 0; r < toks[routes].size; ++r) {
        int route = oj_arr_at(toks, ntoks, routes, r);
        if (route < 0 || toks[route].type != OJ_OBJECT) continue;
        int pos = oj_obj_get(js, toks, ntoks, route, "positive");
        if (pos < 0 || toks[pos].type != OJ_ARRAY) continue;
        int skip = oj_obj_get(js, toks, ntoks, route, "exclude");
        if (skip >= 0 && toks[skip].type == OJ_ARRAY && any_keyword(js, toks, ntoks, skip, lowered)) continue;
        float score = embed_best(js, toks, ntoks, pos, m, q, qn);
        int avoid = oj_obj_get(js, toks, ntoks, route, "avoid");
        if (avoid >= 0 && toks[avoid].type == OJ_ARRAY) {
            float a = embed_best(js, toks, ntoks, avoid, m, q, qn);
            if (a > 0) score -= (float)weight * a;
        }
        int keywords = oj_obj_get(js, toks, ntoks, route, "keywords");
        if (keywords >= 0 && toks[keywords].type == OJ_ARRAY && any_keyword(js, toks, ntoks, keywords, lowered)) {
            score += 0.15f; /* an explicit keyword is evidence, not a verdict */
        }
        int min_token = oj_obj_get(js, toks, ntoks, route, "min_score");
        double min_score = min_token >= 0 ? oj_number(js, &toks[min_token], 0.45) : 0.45;
        if (score >= (float)min_score && score > best_score) {
            best = route;
            best_score = score;
        }
    }
    return best;
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
    int suffix = oj_obj_get(js, toks, ntoks, route, "suffix");
    if (suffix >= 0 && toks[suffix].type == OJ_STRING) {
        oj_unescape(js, &toks[suffix], pick->suffix, sizeof pick->suffix);
    }
    int negative = oj_obj_get(js, toks, ntoks, route, "negative");
    if (negative >= 0 && toks[negative].type == OJ_STRING) {
        oj_unescape(js, &toks[negative], pick->negative, sizeof pick->negative);
    }
    int prompt = oj_obj_get(js, toks, ntoks, route, "prompt");
    if (prompt >= 0 && toks[prompt].type == OJ_STRING) {
        oj_unescape(js, &toks[prompt], pick->prompt, sizeof pick->prompt);
    }
    int steps = oj_obj_get(js, toks, ntoks, route, "steps");
    if (steps >= 0) {
        double v = oj_number(js, &toks[steps], 0);
        if (v >= 1 && v <= 100) pick->steps = (int)v;
    }
    int guidance = oj_obj_get(js, toks, ntoks, route, "guidance");
    if (guidance >= 0) {
        double v = oj_number(js, &toks[guidance], 0);
        if (v >= 0 && v <= 30) pick->guidance = (float)v;
    }
    int loras = oj_obj_get(js, toks, ntoks, route, "loras");
    if (loras < 0 || toks[loras].type != OJ_ARRAY) return pick->suffix[0] || pick->prefix[0] || pick->prompt[0];
    for (int i = 0; i < toks[loras].size && pick->lora_count < OLORA_ROUTE_MAX_LORAS; ++i) {
        int item = oj_arr_at(toks, ntoks, loras, i);
        if (item < 0 || toks[item].type != OJ_OBJECT) return false;
        int id = oj_obj_get(js, toks, ntoks, item, "id");
        bool is_path = false;
        if (id < 0) {
            id = oj_obj_get(js, toks, ntoks, item, "path");
            is_path = true;
        }
        if (id < 0 || toks[id].type != OJ_STRING) return false;
        char *slot = pick->ids[pick->lora_count];
        if (!oj_unescape(js, &toks[id], slot, sizeof pick->ids[0]) ||
            !(is_path ? safe_path(slot) : safe_id(slot))) return false;
        double scale = 1.0;
        int scale_token = oj_obj_get(js, toks, ntoks, item, "scale");
        if (scale_token >= 0) scale = oj_number(js, &toks[scale_token], NAN);
        if (!isfinite(scale) || scale < -4.0 || scale > 4.0) return false;
        pick->scales[pick->lora_count++] = (float)scale;
    }
    return pick->lora_count > 0 || pick->suffix[0] || pick->prefix[0] || pick->prompt[0];
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
    const cbed_model *embed = embed_model();
    if (embed && routes >= 0 && toks[routes].type == OJ_ARRAY && lowered) {
        int route = embed_pick_route(json, toks, ntoks, routes, prompt, lowered, embed);
        if (route >= 0) found = fill_pick(json, toks, ntoks, route, pick);
    }
    if (!found && routes >= 0 && toks[routes].type == OJ_ARRAY && lowered) {
        for (int r = 0; r < toks[routes].size && !found; ++r) {
            int route = oj_arr_at(toks, ntoks, routes, r);
            if (route < 0 || toks[route].type != OJ_OBJECT) continue;
            if (embed && oj_obj_get(json, toks, ntoks, route, "positive") >= 0) continue; /* scored above */
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

static char *read_routes_file(size_t *len_out) {
    const char *path = getenv("OMNISERVE_NATIVE_LORA_ROUTES");
    if (!path || !path[0]) return NULL;
    FILE *file = fopen(path, "rb");
    if (!file) return NULL;
    char *data = malloc(256 * 1024 + 1);
    size_t len = data ? fread(data, 1, 256 * 1024 + 1, file) : 0;
    fclose(file);
    if (!data || len > 256 * 1024) {
        free(data);
        return NULL;
    }
    data[len] = 0;
    *len_out = len;
    return data;
}

bool olora_route_select_task_json(const char *json, size_t json_len, const char *task, olora_route_pick *pick) {
    if (!json || !task || !task[0] || !pick) return false;
    int cap = json_len + 2 < 16384 ? (int)json_len + 2 : 16384;
    oj_tok *toks = calloc((size_t)cap, sizeof *toks);
    if (!toks) return false;
    int ntoks = oj_parse(json, json_len, toks, cap);
    int routes = ntoks > 0 && toks[0].type == OJ_OBJECT ? oj_obj_get(json, toks, ntoks, 0, "edit_routes") : -1;
    bool found = false;
    if (routes >= 0 && toks[routes].type == OJ_ARRAY) {
        for (int r = 0; r < toks[routes].size && !found; ++r) {
            int route = oj_arr_at(toks, ntoks, routes, r);
            int name = route >= 0 && toks[route].type == OJ_OBJECT ? oj_obj_get(json, toks, ntoks, route, "task") : -1;
            char value[64];
            if (name < 0 || toks[name].type != OJ_STRING || !oj_unescape(json, &toks[name], value, sizeof value)) continue;
            if (strcmp(value, task) != 0) continue;
            found = fill_pick(json, toks, ntoks, route, pick);
            if (found && !pick->name[0]) snprintf(pick->name, sizeof pick->name, "%s", value);
        }
    }
    free(toks);
    if (!found) memset(pick, 0, sizeof *pick);
    return found;
}

bool olora_route_select_task(const char *task, olora_route_pick *pick) {
    size_t len = 0;
    char *data = read_routes_file(&len);
    if (!data) return false;
    bool ok = olora_route_select_task_json(data, len, task, pick);
    free(data);
    return ok;
}

bool olora_route_select(const char *prompt, olora_route_pick *pick) {
    size_t len = 0;
    char *data = read_routes_file(&len);
    if (!data) return false;
    bool ok = olora_route_select_json(data, len, prompt, pick);
    free(data);
    return ok;
}
