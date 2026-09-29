#include "obackend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(expr) do { if (!(expr)) { fprintf(stderr, "failed: %s\n", #expr); return 1; } } while (0)

int main(void) {
    const char *red_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC";
    oimg_req req = {0};
    CHECK(osd_prepare_image(&req));
    req.image_base64 = strdup(red_png);
    CHECK(osd_prepare_image(&req));
    CHECK(req.image_width == 1 && req.image_height == 1);
    CHECK(req.image_pixels[0] == 255 && req.image_pixels[1] == 0 && req.image_pixels[2] == 0);
    free(req.image_pixels);
    free(req.image_base64);
    const char *invalid[] = {"", "abc", "!!!!", "=AAA", "A=AA", "AA=A", "AA==AAAA",
                             "AB==", "AAB=", "YWJj", "data:image/png;base64,YWJj"};
    for (size_t i = 0; i < sizeof invalid / sizeof invalid[0]; ++i) {
        memset(&req, 0, sizeof req);
        req.image_base64 = strdup(invalid[i]);
        CHECK(!osd_prepare_image(&req));
        CHECK(!req.image_pixels);
        free(req.image_base64);
    }
    puts("PNG decode and malformed base64 checks passed");
    return 0;
}
