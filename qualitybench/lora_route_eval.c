/* Route-accuracy harness: gcc -O2 -Iinclude qualitybench/lora_route_eval.c src/olora_route.c src/cbed.c src/ojson.c -lm -lpthread
 * usage: lora_route_eval routes.json prompts.tsv   (prompts.tsv: expected_route<TAB>prompt; "-" = no route) */
#include "olora_route.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 3) return 2;
    FILE *rf = fopen(argv[1], "rb");
    if (!rf) return 2;
    char *js = malloc(1 << 20);
    size_t jl = fread(js, 1, (1 << 20) - 1, rf);
    fclose(rf);
    js[jl] = 0;
    FILE *pf = fopen(argv[2], "r");
    if (!pf) return 2;
    char line[1024];
    int total = 0, ok = 0;
    while (fgets(line, sizeof line, pf)) {
        char *tab = strchr(line, '\t');
        if (!tab) continue;
        *tab = 0;
        char *prompt = tab + 1;
        prompt[strcspn(prompt, "\r\n")] = 0;
        olora_route_pick pick;
        bool hit = olora_route_select_json(js, jl, prompt, &pick);
        const char *got = hit ? pick.name : "-";
        int good = strcmp(got, line) == 0;
        total++;
        ok += good;
        if (!good) printf("MISS expected=%-13s got=%-13s | %s\n", line, got, prompt);
    }
    printf("%d/%d correct\n", ok, total);
    return ok == total ? 0 : 1;
}
