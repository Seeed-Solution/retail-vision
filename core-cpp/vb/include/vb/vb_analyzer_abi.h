/* Application analyzer plugin C ABI (spec BASE-1 §6.2, M1.6). */
#ifndef VB_ANALYZER_ABI_H
#define VB_ANALYZER_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VB_ANALYZER_ABI 1

typedef struct vb_track {
    uint32_t track_id;
    float cx, cy, w, h, score;
    int32_t class_id;
    float vx, vy;
    uint32_t hits, misses;
    const float* kpts; /* kpt_count * 3: x, y, conf */
    uint32_t kpt_count;
} vb_track;

typedef struct vb_frame_meta {
    uint32_t stream_index;
    uint64_t seq;
    double wall_ms, t_mono_s;
    int32_t src_w, src_h, model_w, model_h;
    float scale, pad_x, pad_y;
    uint8_t align; /* 0 = center, 1 = top_left */
} vb_frame_meta;

typedef void (*vb_emit_fn)(void* sink, const char* type, uint32_t track_id, const char* fields_json);

typedef struct vb_analyzer_api {
    uint32_t abi;
    const char* name;
    uint32_t attr_count;
    const char* const* attr_names;
    void* (*create)(const char* config_json, char* err, size_t errlen);
    void (*destroy)(void* self);
    int (*on_frame)(void* self, const vb_frame_meta* m, const vb_track* tracks, size_t n,
                    float* attrs_out, vb_emit_fn emit, void* sink);
    void (*on_track_removed)(void* self, uint32_t track_id, double t_mono_s);
} vb_analyzer_api;

/* The only symbol a plugin must export. */
const vb_analyzer_api* vb_analyzer_entry(void);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* VB_ANALYZER_ABI_H */
