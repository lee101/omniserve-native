#include "oimage.h"

#include "ojson.h"

#include "olora_route.h"
#include "onsfw.h"
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static void set_error(char *error, size_t cap, const char *message) {
    if (error && cap) snprintf(error, cap, "%s", message);
}

void oimage_request_init(oimage_request *request) {
    memset(request, 0, sizeof *request);
    request->generation.width = 1024;
    request->generation.height = 1024;
    request->generation.steps = 9;
    request->generation.batch_count = 1;
    request->generation.guidance_scale = 0.0f;
    request->generation.seed = 0;
    request->generation.teleport = false;
    request->generation.teleport_start_step = 0;
    request->generation.strength = 0.6f;
    request->count = 1;
}

static int image_env_int(const char *name, int fallback, int minimum, int maximum) {
    const char *value = getenv(name);
    if (!value || !value[0]) return fallback;
    char *end = NULL;
    errno = 0;
    long parsed = strtol(value, &end, 10);
    if (errno || end == value || *end || parsed < minimum || parsed > maximum) return fallback;
    return (int)parsed;
}

int oimage_gpu_headroom_mb(void) {
    return image_env_int("OMNISERVE_NATIVE_SD_MIN_FREE_MB", 4096, 0, 65536);
}

bool oimage_gpu_headroom_ok(double free_gib, char *error, size_t error_cap) {
    int minimum_mb = oimage_gpu_headroom_mb();
    if (minimum_mb == 0 || free_gib < 0.0 || free_gib * 1024.0 >= (double)minimum_mb) {
        return true;
    }
    if (error && error_cap) {
        snprintf(error, error_cap,
                 "image generation needs at least %d MB free GPU memory; only %.0f MB is free",
                 minimum_mb, free_gib * 1024.0);
    }
    return false;
}

static bool safe_lora_id(const char *value) {
    if (!value || !value[0] || strlen(value) > 128) return false;
    for (const unsigned char *p = (const unsigned char *)value; *p; ++p) {
        if (!((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') ||
              (*p >= '0' && *p <= '9') || *p == '-' || *p == '_' || *p == '.')) {
            return false;
        }
    }
    return strcmp(value, ".") != 0 && strcmp(value, "..") != 0;
}

static bool safe_lora_filename(const char *value) {
    if (!value || !value[0] || strlen(value) > 255 || value[0] == '.' ||
        strchr(value, '/') || strchr(value, '\\') || strstr(value, "..")) return false;
    size_t len = strlen(value);
    static const char suffix[] = ".safetensors";
    return len > sizeof suffix - 1 &&
           strcmp(value + len - (sizeof suffix - 1), suffix) == 0;
}

static bool resolve_cached_lora_file(const char *directory, const char *filename,
                                     char **out_path) {
    size_t needed = strlen(directory) + strlen(filename) + 2;
    char *path = malloc(needed);
    if (!path) return false;
    snprintf(path, needed, "%s/%s", directory, filename);
    struct stat st;
    if (stat(path, &st) != 0 || !S_ISREG(st.st_mode)) {
        free(path);
        return false;
    }
    char resolved_directory[PATH_MAX];
    char resolved_path[PATH_MAX];
    if (!realpath(directory, resolved_directory) || !realpath(path, resolved_path)) {
        free(path);
        return false;
    }
    size_t directory_len = strlen(resolved_directory);
    if (strncmp(resolved_path, resolved_directory, directory_len) != 0 ||
        resolved_path[directory_len] != '/') {
        free(path);
        return false;
    }
    free(path);
    *out_path = strdup(resolved_path);
    return *out_path != NULL;
}

static char *default_lora_registry_path(const char *directory) {
    const char *slash = strrchr(directory, '/');
    if (!slash || slash == directory) return NULL;
    size_t parent_len = (size_t)(slash - directory);
    const char suffix[] = "/lora_registry.json";
    char *path = malloc(parent_len + sizeof suffix);
    if (!path) return NULL;
    memcpy(path, directory, parent_len);
    memcpy(path + parent_len, suffix, sizeof suffix);
    return path;
}

static char *read_small_file(const char *path, size_t *out_len) {
    FILE *file = fopen(path, "rb");
    if (!file) return NULL;
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return NULL;
    }
    long size = ftell(file);
    if (size < 0 || size > 1024L * 1024L || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return NULL;
    }
    char *data = malloc((size_t)size + 1);
    if (!data) {
        fclose(file);
        return NULL;
    }
    size_t read_len = fread(data, 1, (size_t)size, file);
    fclose(file);
    if (read_len != (size_t)size) {
        free(data);
        return NULL;
    }
    data[read_len] = 0;
    *out_len = read_len;
    return data;
}

static char *registry_lora_filename(const char *directory, const char *id) {
    const char *configured = getenv("OMNISERVE_NATIVE_LORA_REGISTRY");
    char *fallback = NULL;
    const char *registry_path = configured && configured[0] ? configured : NULL;
    if (!registry_path) {
        fallback = default_lora_registry_path(directory);
        registry_path = fallback;
    }
    if (!registry_path) return NULL;

    size_t json_len = 0;
    char *json = read_small_file(registry_path, &json_len);
    free(fallback);
    if (!json) return NULL;

    int token_cap = 65536;
    oj_tok *tokens = calloc((size_t)token_cap, sizeof *tokens);
    if (!tokens) {
        free(json);
        return NULL;
    }
    int token_count = oj_parse(json, json_len, tokens, token_cap);
    if (token_count <= 0 || tokens[0].type != OJ_ARRAY) {
        free(tokens);
        free(json);
        return NULL;
    }

    char *filename = NULL;
    for (int i = 1; i < token_count; i++) {
        if (tokens[i].parent != 0 || tokens[i].type != OJ_OBJECT) continue;
        int id_token = oj_obj_get(json, tokens, token_count, i, "id");
        if (id_token < 0 || !oj_str_eq(json, &tokens[id_token], id)) continue;
        int path_token = oj_obj_get(json, tokens, token_count, i, "path");
        if (path_token < 0 || tokens[path_token].type != OJ_STRING) break;
        char *path = oj_strdup(json, &tokens[path_token]);
        if (!path) break;
        const char *base = strrchr(path, '/');
        base = base ? base + 1 : path;
        if (safe_lora_filename(base)) filename = strdup(base);
        free(path);
        break;
    }

    free(tokens);
    free(json);
    return filename;
}

static char *cached_lora_path(const char *id, const char *filename,
                              char *error, size_t error_cap) {
    const char *directory = getenv("OMNISERVE_NATIVE_LORA_DIR");
    if (!directory || !directory[0]) {
        set_error(error, error_cap, "lora_id requires OMNISERVE_NATIVE_LORA_DIR");
        return NULL;
    }
    if (!safe_lora_id(id)) {
        set_error(error, error_cap, "lora_id contains unsupported characters");
        return NULL;
    }
    char fallback[144];
    if (!filename) {
        snprintf(fallback, sizeof fallback, "%s.safetensors", id);
        filename = fallback;
    }
    if (!safe_lora_filename(filename)) {
        set_error(error, error_cap, "lora_filename must be a safe .safetensors basename");
        return NULL;
    }
    char *path = NULL;
    if (resolve_cached_lora_file(directory, filename, &path)) return path;
    if (filename == fallback) {
        char *registry_filename = registry_lora_filename(directory, id);
        if (registry_filename) {
            bool ok = resolve_cached_lora_file(directory, registry_filename, &path);
            free(registry_filename);
            if (ok) return path;
        }
    }
    set_error(error, error_cap, "lora_id is not installed in the native cache");
    return NULL;
}

void oimage_request_free(oimage_request *request) {
    if (!request) return;
    free(request->prompt);
    free(request->negative_prompt);
    free(request->generation.image_base64);
    free(request->generation.image_pixels);
    if (request->loras) {
        for (size_t i = 0; i < request->generation.lora_count; ++i) {
            free((char *)request->loras[i].path);
        }
    }
    free(request->loras);
    memset(request, 0, sizeof *request);
}

static bool add_automatic_nsfw_lora(oimage_request *request) {
    if (!request || request->generation.lora_count ||
        !onsfw_prompt_has_word(request->prompt)) {
        return true;
    }
    const char *path = getenv("OMNISERVE_NATIVE_NSFW_LORA_PATH");
    if (!path || path[0] != '/' || access(path, R_OK) != 0) {
        if (path && path[0]) {
            fprintf(stderr, "[safety] prompt word match but NSFW LoRA is unavailable: %s\n",
                    path);
        }
        return true;
    }
    char *owned_path = strdup(path);
    if (!owned_path) return false;
    size_t count = request->generation.lora_count;
    oimg_lora *grown = realloc(request->loras, (count + 1) * sizeof *grown);
    if (!grown) {
        free(owned_path);
        return false;
    }
    double scale = 0.6;
    const char *scale_text = getenv("OMNISERVE_NATIVE_NSFW_LORA_SCALE");
    if (scale_text && scale_text[0]) {
        char *end = NULL;
        double parsed = strtod(scale_text, &end);
        if (end != scale_text && isfinite(parsed) && parsed >= -4.0 && parsed <= 4.0) {
            scale = parsed;
        }
    }
    grown[count].path = owned_path;
    grown[count].scale = (float)scale;
    request->loras = grown;
    request->generation.loras = grown;
    request->generation.lora_count = count + 1;
    fprintf(stderr, "[safety] prompt word match: enabling NSFW LoRA scale=%.2f\n", scale);
    return true;
}
static bool parse_size(const char *json, const oj_tok *token, int *width, int *height) {
    if (token->type != OJ_STRING || token->end <= token->start) return false;
    size_t len = (size_t)(token->end - token->start);
    if (len >= 64) return false;
    char value[64];
    memcpy(value, json + token->start, len);
    value[len] = 0;
    char *sep = strchr(value, 'x');
    if (!sep) sep = strchr(value, 'X');
    if (!sep || sep == value || !sep[1]) return false;
    *sep = 0;
    char *end_w = NULL;
    char *end_h = NULL;
    errno = 0;
    long w = strtol(value, &end_w, 10);
    long h = strtol(sep + 1, &end_h, 10);
    if (errno || !end_w || *end_w || !end_h || *end_h ||
        w < 64 || h < 64 || w > 4096 || h > 4096 ||
        (w % 64) != 0 || (h % 64) != 0) return false;
    *width = (int)w;
    *height = (int)h;
    return true;
}

static bool token_int(const char *json, const oj_tok *token, int *value) {
    if (token->type != OJ_PRIMITIVE || token->end <= token->start) return false;
    size_t len = (size_t)(token->end - token->start);
    if (len >= 64) return false;
    char bounded[64];
    memcpy(bounded, json + token->start, len);
    bounded[len] = 0;
    char *end = NULL;
    errno = 0;
    long parsed = strtol(bounded, &end, 10);
    if (errno || end != bounded + len || parsed < INT_MIN || parsed > INT_MAX) return false;
    *value = (int)parsed;
    return true;
}

static bool token_int64(const char *json, const oj_tok *token, int64_t *value) {
    if (token->type != OJ_PRIMITIVE || token->end <= token->start) return false;
    size_t len = (size_t)(token->end - token->start);
    if (len >= 64) return false;
    char bounded[64];
    memcpy(bounded, json + token->start, len);
    bounded[len] = 0;
    char *end = NULL;
    errno = 0;
    long long parsed = strtoll(bounded, &end, 10);
    if (errno || end != bounded + len || parsed < INT64_MIN || parsed > INT64_MAX) return false;
    *value = (int64_t)parsed;
    return true;
}

static bool token_finite_double(const char *json, const oj_tok *token, double *value) {
    if (token->type != OJ_PRIMITIVE || token->end <= token->start) return false;
    size_t len = (size_t)(token->end - token->start);
    if (len >= 128) return false;
    char bounded[128];
    memcpy(bounded, json + token->start, len);
    bounded[len] = 0;
    char *end = NULL;
    errno = 0;
    double parsed = strtod(bounded, &end);
    if (errno || end != bounded + len || !isfinite(parsed)) return false;
    *value = parsed;
    return true;
}

static bool token_bool(const char *json, const oj_tok *token, bool *value) {
    if (token->type != OJ_PRIMITIVE || token->end <= token->start) return false;
    size_t len = (size_t)(token->end - token->start);
    if (len == 4 && memcmp(json + token->start, "true", 4) == 0) {
        *value = true;
        return true;
    }
    if (len == 5 && memcmp(json + token->start, "false", 5) == 0) {
        *value = false;
        return true;
    }
    return false;
}

static bool add_routed_loras(oimage_request *request, const char *json,
                             const oj_tok *tokens, int token_count) {
    if (!olora_route_enabled() || request->generation.lora_count ||
        request->generation.image_base64 || !request->prompt) {
        return true;
    }
    int token = oj_obj_get(json, tokens, token_count, 0, "auto_lora");
    bool wanted = true;
    if (token >= 0 && token_bool(json, &tokens[token], &wanted) && !wanted) return true;
    olora_route_pick pick;
    if (!olora_route_select(request->prompt, &pick)) return true;
    oimg_lora *loras = calloc(pick.lora_count, sizeof *loras);
    if (!loras) return false;
    char error[160];
    for (size_t i = 0; i < pick.lora_count; ++i) {
        loras[i].path = cached_lora_path(pick.ids[i], NULL, error, sizeof error);
        loras[i].scale = pick.scales[i];
        if (!loras[i].path) {
            fprintf(stderr, "[lora-route] %s: %s: %s; generating without route\n",
                    pick.name, pick.ids[i], error);
            for (size_t j = 0; j < i; ++j) free((char *)loras[j].path);
            free(loras);
            return true;
        }
    }
    if (pick.prefix[0]) {
        size_t prefix_len = strlen(pick.prefix);
        size_t prompt_len = strlen(request->prompt);
        char *prompt = malloc(prefix_len + prompt_len + 1);
        if (!prompt) {
            for (size_t i = 0; i < pick.lora_count; ++i) free((char *)loras[i].path);
            free(loras);
            return false;
        }
        memcpy(prompt, pick.prefix, prefix_len);
        memcpy(prompt + prefix_len, request->prompt, prompt_len + 1);
        free(request->prompt);
        request->prompt = prompt;
        request->generation.prompt = prompt;
    }
    free(request->loras);
    request->loras = loras;
    request->generation.loras = loras;
    request->generation.lora_count = pick.lora_count;
    snprintf(request->lora_route, sizeof request->lora_route, "%s", pick.name);
    fprintf(stderr, "[lora-route] %s: %zu lora(s), first %s@%.2f\n", pick.name,
            pick.lora_count, pick.ids[0], (double)pick.scales[0]);
    return true;
}

bool oimage_request_parse(const char *json, size_t json_len, oimage_request *request,
                          char *error, size_t error_cap) {
    if (!json || !request) {
        set_error(error, error_cap, "invalid image request");
        return false;
    }
    oimage_request_init(request);
    enum { MAX_IMAGE_TOKENS = 256 };
    oj_tok tokens[MAX_IMAGE_TOKENS];
    int token_count = oj_parse(json, json_len, tokens, MAX_IMAGE_TOKENS);
    if (token_count <= 0 || tokens[0].type != OJ_OBJECT) {
        set_error(error, error_cap, "invalid JSON body");
        return false;
    }

    int token = oj_obj_get(json, tokens, token_count, 0, "prompt");
    if (token < 0 || tokens[token].type != OJ_STRING) {
        set_error(error, error_cap, "prompt required");
        return false;
    }
    request->prompt = oj_strdup(json, &tokens[token]);
    if (!request->prompt || !request->prompt[0]) {
        set_error(error, error_cap, request->prompt ? "prompt required" : "allocation failed");
        oimage_request_free(request);
        return false;
    }
    request->generation.prompt = request->prompt;

    token = oj_obj_get(json, tokens, token_count, 0, "negative_prompt");
    if (token >= 0 && tokens[token].type != OJ_STRING) {
        set_error(error, error_cap, "negative_prompt must be a string");
        oimage_request_free(request);
        return false;
    }
    if (token >= 0) {
        request->negative_prompt = oj_strdup(json, &tokens[token]);
        if (!request->negative_prompt) {
            set_error(error, error_cap, "allocation failed");
            oimage_request_free(request);
            return false;
        }
        request->generation.negative_prompt = request->negative_prompt;
    }

    token = oj_obj_get(json, tokens, token_count, 0, "size");
    if (token >= 0 && !parse_size(json, &tokens[token],
                                  &request->generation.width,
                                  &request->generation.height)) {
        set_error(error, error_cap, "size must be WIDTHxHEIGHT in 64-pixel increments");
        oimage_request_free(request);
        return false;
    }
    token = oj_obj_get(json, tokens, token_count, 0, "width");
    if (token >= 0 && !token_int(json, &tokens[token], &request->generation.width)) {
        set_error(error, error_cap, "width must be an integer");
        oimage_request_free(request);
        return false;
    }
    token = oj_obj_get(json, tokens, token_count, 0, "height");
    if (token >= 0 && !token_int(json, &tokens[token], &request->generation.height)) {
        set_error(error, error_cap, "height must be an integer");
        oimage_request_free(request);
        return false;
    }
    if (request->generation.width < 64 || request->generation.height < 64 ||
        request->generation.width > 4096 || request->generation.height > 4096 ||
        request->generation.width % 64 || request->generation.height % 64) {
        set_error(error, error_cap, "width and height must be 64..4096 in 64-pixel increments");
        oimage_request_free(request);
        return false;
    }

    int steps = request->generation.steps;
    token = oj_obj_get(json, tokens, token_count, 0, "steps");
    if (token < 0) token = oj_obj_get(json, tokens, token_count, 0, "num_inference_steps");
    if (token >= 0 && !token_int(json, &tokens[token], &steps)) {
        set_error(error, error_cap, "steps must be an integer");
        oimage_request_free(request);
        return false;
    }
    if (steps < 1 || steps > 100) {
        set_error(error, error_cap, "steps must be between 1 and 100");
        oimage_request_free(request);
        return false;
    }
    request->generation.steps = steps;
    token = oj_obj_get(json, tokens, token_count, 0, "guidance_scale");
    if (token >= 0) {
        double guidance = 0.0;
        if (!token_finite_double(json, &tokens[token], &guidance)) {
            set_error(error, error_cap, "guidance_scale must be a finite number");
            oimage_request_free(request);
            return false;
        }
        request->generation.guidance_scale = (float)guidance;
    }
    if (request->generation.guidance_scale < 0.0f || request->generation.guidance_scale > 30.0f) {
        set_error(error, error_cap, "guidance_scale must be between 0 and 30");
        oimage_request_free(request);
        return false;
    }
    token = oj_obj_get(json, tokens, token_count, 0, "seed");
    if (token >= 0 && !token_int64(json, &tokens[token], &request->generation.seed)) {
        set_error(error, error_cap, "seed must be a 64-bit integer");
        oimage_request_free(request);
        return false;
    }
    token = oj_obj_get(json, tokens, token_count, 0, "cache");
    if (token >= 0 && (!token_bool(json, &tokens[token], &request->generation.cache) ||
                      (request->generation.cache && request->generation.seed < 0))) {
        set_error(error, error_cap, "cache must be a boolean and requires a nonnegative seed");
        oimage_request_free(request);
        return false;
    }
    token = oj_obj_get(json, tokens, token_count, 0, "teleport");
    if (token >= 0 && (!token_bool(json, &tokens[token], &request->generation.teleport) ||
                      (request->generation.teleport && request->generation.seed < 0))) {
        set_error(error, error_cap, "teleport must be a boolean and requires a nonnegative seed");
        oimage_request_free(request);
        return false;
    }
    token = oj_obj_get(json, tokens, token_count, 0, "teleport_start_step");
    if (token >= 0) {
        if (!token_int(json, &tokens[token], &request->generation.teleport_start_step)) {
            set_error(error, error_cap, "teleport_start_step must be an integer");
            oimage_request_free(request);
            return false;
        }
        if (request->generation.teleport_start_step < 1 ||
            request->generation.teleport_start_step >= request->generation.steps) {
            set_error(error, error_cap, "teleport_start_step must be between 1 and steps-1");
            oimage_request_free(request);
            return false;
        }
    }
    token = oj_obj_get(json, tokens, token_count, 0, "loras");
    if (token >= 0) {
        if (tokens[token].type != OJ_ARRAY || tokens[token].size > 8) {
            set_error(error, error_cap, "loras must be an array with at most 8 entries");
            oimage_request_free(request);
            return false;
        }
        size_t count = (size_t)tokens[token].size;
        request->loras = calloc(count, sizeof *request->loras);
        if (count && !request->loras) {
            set_error(error, error_cap, "allocation failed");
            oimage_request_free(request);
            return false;
        }
        for (size_t i = 0; i < count; ++i) {
            int item = oj_arr_at(tokens, token_count, token, (int)i);
            int path_token = -1;
            int scale_token = -1;
            if (item >= 0 && tokens[item].type == OJ_STRING) {
                path_token = item;
            } else if (item >= 0 && tokens[item].type == OJ_OBJECT) {
                path_token = oj_obj_get(json, tokens, token_count, item, "path");
                scale_token = oj_obj_get(json, tokens, token_count, item, "scale");
            }
            if (path_token < 0 || tokens[path_token].type != OJ_STRING) {
                set_error(error, error_cap, "each lora must be a path string or object with path");
                oimage_request_free(request);
                return false;
            }
            request->loras[i].path = oj_strdup(json, &tokens[path_token]);
            request->loras[i].scale = 1.0f;
            if (!request->loras[i].path) {
                set_error(error, error_cap, "allocation failed");
                oimage_request_free(request);
                return false;
            }
            if (scale_token >= 0) {
                double scale = 0.0;
                if (!token_finite_double(json, &tokens[scale_token], &scale) ||
                    scale < -4.0 || scale > 4.0) {
                    set_error(error, error_cap, "lora scale must be between -4 and 4");
                    oimage_request_free(request);
                    return false;
                }
                request->loras[i].scale = (float)scale;
            }
        }
        request->generation.loras = request->loras;
        request->generation.lora_count = count;
        request->direct_lora_paths = count > 0;
    }
    int lora_id_token = oj_obj_get(json, tokens, token_count, 0, "lora_id");
    int lora_filename_token = oj_obj_get(json, tokens, token_count, 0, "lora_filename");
    if (lora_id_token >= 0) {
        if (request->generation.lora_count) {
            set_error(error, error_cap, "use either lora_id or loras, not both");
            oimage_request_free(request);
            return false;
        }
        if (tokens[lora_id_token].type != OJ_STRING) {
            set_error(error, error_cap, "lora_id must be a string");
            oimage_request_free(request);
            return false;
        }
        char *id = oj_strdup(json, &tokens[lora_id_token]);
        if (!id) {
            set_error(error, error_cap, "allocation failed");
            oimage_request_free(request);
            return false;
        }
        char *filename = NULL;
        if (lora_filename_token >= 0) {
            if (tokens[lora_filename_token].type != OJ_STRING) {
                free(id);
                set_error(error, error_cap, "lora_filename must be a string");
                oimage_request_free(request);
                return false;
            }
            filename = oj_strdup(json, &tokens[lora_filename_token]);
            if (!filename) {
                free(id);
                set_error(error, error_cap, "allocation failed");
                oimage_request_free(request);
                return false;
            }
        }
        char *path = cached_lora_path(id, filename, error, error_cap);
        free(filename);
        free(id);
        if (!path) {
            oimage_request_free(request);
            return false;
        }
        request->loras = calloc(1, sizeof *request->loras);
        if (!request->loras) {
            free(path);
            set_error(error, error_cap, "allocation failed");
            oimage_request_free(request);
            return false;
        }
        request->loras[0].path = path;
        request->loras[0].scale = 1.0f;
        request->generation.loras = request->loras;
        request->generation.lora_count = 1;
        int scale_token = oj_obj_get(json, tokens, token_count, 0, "lora_scale");
        if (scale_token >= 0) {
            double scale = 0.0;
            if (!token_finite_double(json, &tokens[scale_token], &scale) ||
                scale < -4.0 || scale > 4.0) {
                set_error(error, error_cap, "lora_scale must be between -4 and 4");
                oimage_request_free(request);
                return false;
            }
            request->loras[0].scale = (float)scale;
        }
    } else if (lora_filename_token >= 0) {
        set_error(error, error_cap, "lora_filename requires lora_id");
        oimage_request_free(request);
        return false;
    }
    if (!add_automatic_nsfw_lora(request)) {
        set_error(error, error_cap, "could not prepare automatic NSFW LoRA");
        oimage_request_free(request);
        return false;
    }
    token = oj_obj_get(json, tokens, token_count, 0, "n");
    if (token >= 0 && !token_int(json, &tokens[token], &request->count)) {
        set_error(error, error_cap, "n must be an integer");
        oimage_request_free(request);
        return false;
    }
    int max_batch = image_env_int("OMNISERVE_NATIVE_SD_MAX_BATCH", 1, 1, 8);
    if (request->count < 1 || request->count > max_batch) {
        set_error(error, error_cap, "n exceeds OMNISERVE_NATIVE_SD_MAX_BATCH");
        oimage_request_free(request);
        return false;
    }
    request->generation.batch_count = request->count;
    token = oj_obj_get(json, tokens, token_count, 0, "image_base64");
    if (token >= 0) {
        if (tokens[token].type != OJ_STRING ||
            !(request->generation.image_base64 = oj_strdup(json, &tokens[token])) ||
            !request->generation.image_base64[0] ||
            strlen(request->generation.image_base64) > (8u << 20)) {
            set_error(error, error_cap, "image_base64 must contain at most 8 MiB of base64 PNG or JPEG");
            oimage_request_free(request);
            return false;
        }
        if (request->generation.teleport) {
            set_error(error, error_cap, "latent teleportation is not supported for img2img");
            oimage_request_free(request);
            return false;
        }
    }
    token = oj_obj_get(json, tokens, token_count, 0, "strength");
    if (token >= 0) {
        double strength;
        if (!request->generation.image_base64 ||
            !token_finite_double(json, &tokens[token], &strength) ||
            strength <= 0.0 || strength > 1.0) {
            set_error(error, error_cap, "strength requires image_base64 and must be in (0, 1]");
            oimage_request_free(request);
            return false;
        }
        request->generation.strength = (float)strength;
    }
    if (!add_routed_loras(request, json, tokens, token_count)) {
        set_error(error, error_cap, "could not prepare routed LoRA");
        oimage_request_free(request);
        return false;
    }
    return true;
}

static const char B64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static size_t base64_size(size_t input) {
    if (input > (SIZE_MAX - 2) / 3) return 0;
    return ((input + 2) / 3) * 4;
}

static void base64_encode(const unsigned char *input, size_t len, char *output) {
    size_t in = 0;
    size_t out = 0;
    while (in + 3 <= len) {
        unsigned value = ((unsigned)input[in] << 16) |
                         ((unsigned)input[in + 1] << 8) | input[in + 2];
        output[out++] = B64[(value >> 18) & 63];
        output[out++] = B64[(value >> 12) & 63];
        output[out++] = B64[(value >> 6) & 63];
        output[out++] = B64[value & 63];
        in += 3;
    }
    if (in < len) {
        unsigned value = (unsigned)input[in] << 16;
        output[out++] = B64[(value >> 18) & 63];
        if (in + 1 < len) {
            value |= (unsigned)input[in + 1] << 8;
            output[out++] = B64[(value >> 12) & 63];
            output[out++] = B64[(value >> 6) & 63];
            output[out++] = '=';
        } else {
            output[out++] = B64[(value >> 12) & 63];
            output[out++] = '=';
            output[out++] = '=';
        }
    }
}

bool oimage_openai_response(const oimg_result *result, const char *model, long long seed,
                            char **json_out, size_t *json_len_out) {
    if (!result || !json_out || !json_len_out) return false;
    size_t count = result->image_count ? result->image_count : 1;
    if (count > 8) return false;
    size_t encoded_total = 0;
    for (size_t i = 0; i < count; ++i) {
        const unsigned char *image = result->image_count ? result->images[i] : result->png;
        size_t image_len = result->image_count ? result->image_lens[i] : result->png_len;
        size_t encoded_len = base64_size(image_len);
        if (!image || !image_len || !encoded_len || encoded_total > SIZE_MAX - encoded_len) return false;
        encoded_total += encoded_len;
    }
    const char *safe_model = model && model[0] ? model : "diffusion";
    const char *format = result->format ? result->format : "png";
    if (encoded_total > SIZE_MAX - 4096 - count * 384) return false;
    size_t capacity = encoded_total + 4096 + count * 384;
    char *json = malloc(capacity);
    if (!json) return false;
    int wrote = snprintf(json, capacity,
        "{\"created\":%lld,\"model\":\"%s\",\"format\":\"%s\",\"data\":[",
        (long long)time(NULL), safe_model, format);
    if (wrote < 0 || (size_t)wrote >= capacity) { free(json); return false; }
    size_t used = (size_t)wrote;
    for (size_t i = 0; i < count; ++i) {
        const unsigned char *image = result->image_count ? result->images[i] : result->png;
        size_t image_len = result->image_count ? result->image_lens[i] : result->png_len;
        size_t encoded_len = base64_size(image_len);
        wrote = snprintf(json + used, capacity - used,
            "%s{\"b64_json\":\"", i ? "," : "");
        if (wrote < 0 || (size_t)wrote >= capacity - used) { free(json); return false; }
        used += (size_t)wrote;
        base64_encode(image, image_len, json + used);
        used += encoded_len;
        wrote = snprintf(json + used, capacity - used,
            "\",\"seed\":%lld,\"inference_time_ms\":%lld,\"format\":\"%s\"",
            seed + (long long)i, (long long)(result->elapsed_ms + 0.5), format);
        if (wrote < 0 || (size_t)wrote >= capacity - used) { free(json); return false; }
        used += (size_t)wrote;
        if (result->denoiser_cache_threshold > 0.0f) {
            wrote = snprintf(json + used, capacity - used,
                ",\"denoiser_cache\":{\"requested\":\"%s\",\"approximate\":true,\"threshold\":%.6f}",
                result->denoiser_cache_mode ? result->denoiser_cache_mode : "easycache",
                result->denoiser_cache_threshold);
            if (wrote < 0 || (size_t)wrote >= capacity - used) { free(json); return false; }
            used += (size_t)wrote;
        }
        if (result->cache_requested) {
            wrote = snprintf(json + used, capacity - used,
                ",\"cache\":{\"method\":\"exact_request_result\",\"hit\":%s}",
                result->cache_hit ? "true" : "false");
            if (wrote < 0 || (size_t)wrote >= capacity - used) { free(json); return false; }
            used += (size_t)wrote;
        }
        if (result->teleport_requested && count == 1) {
            const char *method = result->teleport_result_cache_hit
                ? "exact_prompt_result_cache"
                : (result->teleport_used
                    ? "exact_prompt_latent_replay" : "full_generation_fallback");
            wrote = snprintf(json + used, capacity - used,
                ",\"teleport\":{\"method\":\"%s\",\"cache_hit\":%s,"
                "\"capture_step\":%d,\"resume_step\":%d}", method,
                result->teleport_cache_hit ? "true" : "false",
                result->teleport_capture_step, result->teleport_resume_step);
            if (wrote < 0 || (size_t)wrote >= capacity - used) { free(json); return false; }
            used += (size_t)wrote;
        }
        if (used + 1 >= capacity) { free(json); return false; }
        json[used++] = '}';
    }
    if (used + 3 > capacity) { free(json); return false; }
    json[used++] = ']';
    json[used++] = '}';
    json[used] = 0;
    *json_out = json;
    *json_len_out = used;
    return true;
}

static bool append_text(char **buf, size_t *len, size_t *cap, const char *text) {
    size_t n = strlen(text);
    if (*len + n + 1 > *cap) {
        size_t next = *cap ? *cap * 2 : 1024;
        while (next < *len + n + 1) next *= 2;
        char *grown = realloc(*buf, next);
        if (!grown) return false;
        *buf = grown;
        *cap = next;
    }
    memcpy(*buf + *len, text, n + 1);
    *len += n;
    return true;
}

static const char *mem_find(const char *hay, size_t hay_len, const char *needle, size_t needle_len) {
    if (!needle_len || hay_len < needle_len) return NULL;
    for (size_t i = 0; i + needle_len <= hay_len; i++)
        if (hay[i] == needle[0] && !memcmp(hay + i, needle, needle_len)) return hay + i;
    return NULL;
}

/* Returns the value of `key=` inside a header value (quoted or bare). */
static bool header_param(const char *value, size_t len, const char *key,
                         const char **out, size_t *out_len) {
    size_t key_len = strlen(key);
    for (size_t i = 0; i + key_len + 1 <= len; i++) {
        if (strncasecmp(value + i, key, key_len) || value[i + key_len] != '=') continue;
        if (i && value[i - 1] != ';' && value[i - 1] != ' ' && value[i - 1] != '\t') continue;
        const char *v = value + i + key_len + 1;
        const char *end = value + len;
        if (v < end && *v == '"') {
            const char *q = memchr(v + 1, '"', (size_t)(end - v - 1));
            if (!q) return false;
            *out = v + 1;
            *out_len = (size_t)(q - v - 1);
        } else {
            const char *e = v;
            while (e < end && *e != ';' && *e != ' ' && *e != '\t' && *e != '\r') e++;
            *out = v;
            *out_len = (size_t)(e - v);
        }
        return *out_len > 0;
    }
    return false;
}

static bool field_is_numeric(const char *name, size_t len) {
    static const char *numeric[] = {"n", "seed", "width", "height", "steps",
                                    "guidance_scale", "strength", "cfg_scale"};
    for (size_t i = 0; i < sizeof numeric / sizeof numeric[0]; i++)
        if (strlen(numeric[i]) == len && !memcmp(numeric[i], name, len)) return true;
    return false;
}

static bool name_is(const char *name, size_t len, const char *want) {
    return strlen(want) == len && !memcmp(name, want, len);
}

char *oimage_edit_multipart_to_json(const char *content_type, size_t content_type_len,
                                    const char *body, size_t body_len, size_t *json_len,
                                    char *model, size_t model_cap,
                                    char *error, size_t error_cap) {
    if (model && model_cap) model[0] = 0;
    const char *boundary = NULL;
    size_t boundary_len = 0;
    if (!content_type || content_type_len < 19 ||
        strncasecmp(content_type, "multipart/form-data", 19) ||
        !header_param(content_type, content_type_len, "boundary", &boundary, &boundary_len) ||
        boundary_len > 200) {
        set_error(error, error_cap, "multipart/form-data with a boundary is required");
        return NULL;
    }
    char delim[208];
    size_t delim_len = (size_t)snprintf(delim, sizeof delim, "--%.*s", (int)boundary_len, boundary);
    char *out = NULL;
    size_t out_len = 0, out_cap = 0;
    bool ok = append_text(&out, &out_len, &out_cap, "{");
    bool have_image = false, have_prompt = false, first = true;
    const char *cursor = mem_find(body, body_len, delim, delim_len);
    while (ok && cursor) {
        cursor += delim_len;
        size_t rest = body_len - (size_t)(cursor - body);
        if (rest >= 2 && cursor[0] == '-' && cursor[1] == '-') break;
        if (rest >= 2 && cursor[0] == '\r' && cursor[1] == '\n') cursor += 2;
        else if (rest >= 1 && cursor[0] == '\n') cursor += 1;
        rest = body_len - (size_t)(cursor - body);
        const char *headers_end = mem_find(cursor, rest, "\r\n\r\n", 4);
        size_t sep = 4;
        if (!headers_end) { headers_end = mem_find(cursor, rest, "\n\n", 2); sep = 2; }
        if (!headers_end) break;
        const char *name = NULL, *filename = NULL;
        size_t name_len = 0, filename_len = 0;
        for (const char *line = cursor; line < headers_end;) {
            const char *eol = memchr(line, '\n', (size_t)(headers_end - line));
            if (!eol) eol = headers_end;
            size_t line_len = (size_t)(eol - line);
            if (line_len >= 20 && !strncasecmp(line, "content-disposition:", 20)) {
                header_param(line, line_len, "name", &name, &name_len);
                if (!header_param(line, line_len, "filename", &filename, &filename_len))
                    filename = NULL;
            }
            line = eol + 1;
        }
        const char *data = headers_end + sep;
        const char *next = mem_find(data, body_len - (size_t)(data - body), delim, delim_len);
        if (!next) break;
        size_t data_len = (size_t)(next - data);
        if (data_len >= 2 && data[data_len - 2] == '\r' && data[data_len - 1] == '\n') data_len -= 2;
        else if (data_len >= 1 && data[data_len - 1] == '\n') data_len -= 1;
        cursor = next;
        if (!name || !name_len) continue;
        bool image_field = name_is(name, name_len, "image") || name_is(name, name_len, "image[]") ||
                           name_is(name, name_len, "image_file");
        if (image_field) {
            if (have_image || !data_len) continue;
            if (data_len > (6u << 20)) {
                set_error(error, error_cap, "image must be at most 6 MiB");
                free(out);
                return NULL;
            }
            size_t encoded = base64_size(data_len);
            ok = append_text(&out, &out_len, &out_cap, first ? "\"image_base64\":\"" : ",\"image_base64\":\"");
            if (ok && out_len + encoded + 2 > out_cap) {
                size_t cap = out_len + encoded + 256;
                char *grown = realloc(out, cap);
                if (!grown) ok = false;
                else { out = grown; out_cap = cap; }
            }
            if (ok) {
                base64_encode((const unsigned char *)data, data_len, out + out_len);
                out_len += encoded;
                out[out_len] = 0;
                ok = append_text(&out, &out_len, &out_cap, "\"");
            }
            have_image = true;
            first = false;
            continue;
        }
        if (filename) continue; /* masks and other files: reference edits are global */
        if (name_is(name, name_len, "image_base64") || name_is(name, name_len, "response_format"))
            continue;
        if (name_is(name, name_len, "model") && model && model_cap) {
            size_t n = data_len < model_cap - 1 ? data_len : model_cap - 1;
            memcpy(model, data, n);
            model[n] = 0;
        }
        if (name_is(name, name_len, "prompt")) have_prompt = data_len > 0;
        ok = append_text(&out, &out_len, &out_cap, first ? "\"" : ",\"");
        if (ok) ok = oj_escape_append(&out, &out_len, &out_cap, name, name_len);
        if (ok) ok = append_text(&out, &out_len, &out_cap, "\":");
        bool numeric = field_is_numeric(name, name_len) && data_len && data_len < 32;
        if (numeric) {
            char tmp[32], *end = NULL;
            memcpy(tmp, data, data_len);
            tmp[data_len] = 0;
            double v = strtod(tmp, &end);
            numeric = end && *end == 0 && isfinite(v);
        }
        if (ok && numeric) {
            char tmp[33];
            memcpy(tmp, data, data_len);
            tmp[data_len] = 0;
            ok = append_text(&out, &out_len, &out_cap, tmp);
        } else if (ok) {
            ok = append_text(&out, &out_len, &out_cap, "\"") &&
                 oj_escape_append(&out, &out_len, &out_cap, data, data_len) &&
                 append_text(&out, &out_len, &out_cap, "\"");
        }
        first = false;
    }
    if (ok) ok = append_text(&out, &out_len, &out_cap, "}");
    if (!ok) {
        free(out);
        set_error(error, error_cap, "out of memory");
        return NULL;
    }
    if (!have_image || !have_prompt) {
        free(out);
        set_error(error, error_cap, "multipart edit requires an image file and a prompt");
        return NULL;
    }
    *json_len = out_len;
    return out;
}
