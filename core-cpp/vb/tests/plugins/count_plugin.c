/* Example analyzer plugin (spec BASE-1 §8 M1.6): per frame, per track,
 * outputs attribute "count.frames" (frames seen by that track) and emits a
 * "count_ten" event when a track reaches 10 frames. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vb/vb_analyzer_abi.h"

#ifdef VB_TEST_BAD_ABI
#define PLUGIN_ABI 99
#else
#define PLUGIN_ABI VB_ANALYZER_ABI
#endif

#define MAX_TRACKS 64

typedef struct {
    uint32_t tid;
    int frames;
} ent_t;

typedef struct {
    ent_t e[MAX_TRACKS];
    size_t n;
} inst_t;

static void* count_create(const char* config_json, char* err, size_t errlen) {
    if (config_json && *config_json && strcmp(config_json, "{}") != 0) {
        if (err && errlen) snprintf(err, errlen, "count: config must be {}");
        return NULL;
    }
    if (err && errlen) err[0] = '\0';
    inst_t* s = (inst_t*)calloc(1, sizeof(inst_t));
    return s;
}

static void count_destroy(void* self) { free(self); }

static int count_on_frame(void* self, const vb_frame_meta* m, const vb_track* tracks,
                          size_t n, float* attrs_out, vb_emit_fn emit, void* sink) {
    (void)m;
    if (!self) return -1;
    inst_t* s = (inst_t*)self;
    for (size_t i = 0; i < n; i++) {
        if (attrs_out) attrs_out[i] = 0.0f;
        ent_t* e = NULL;
        for (size_t k = 0; k < s->n; k++) {
            if (s->e[k].tid == tracks[i].track_id) { e = &s->e[k]; break; }
        }
        if (!e) {
            if (s->n >= MAX_TRACKS) continue;
            e = &s->e[s->n++];
            e->tid = tracks[i].track_id;
            e->frames = 0;
        }
        e->frames++;
        if (attrs_out) attrs_out[i] = (float)e->frames;
        if (e->frames == 10 && emit && sink) {
            emit(sink, "count_ten", tracks[i].track_id, "{\"frames\":10}");
        }
    }
    return 0;
}

static void count_on_track_removed(void* self, uint32_t track_id, double t_mono_s) {
    (void)t_mono_s;
    if (!self) return;
    inst_t* s = (inst_t*)self;
    for (size_t k = 0; k < s->n; k++) {
        if (s->e[k].tid == track_id) {
            s->e[k] = s->e[s->n - 1];
            s->n--;
            return;
        }
    }
}

static const char* const kAttrNames[] = {"count.frames"};

static const vb_analyzer_api kApi = {
    PLUGIN_ABI,
    "count",
    1,
    kAttrNames,
    count_create,
    count_destroy,
    count_on_frame,
    count_on_track_removed,
};

const vb_analyzer_api* vb_analyzer_entry(void) { return &kApi; }
