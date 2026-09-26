// Analyzer chain interface (spec BASE-1 §6.2, M1.6).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "vb/tracker.h"
#include "vb/types.h"

namespace vb {

// B6: stage2 recognition results for this frame; bbox is the crop box in
// source-normalized coords; seq is the frame the read happened on.
struct Stage2Read {
    uint32_t track_id = 0;
    std::string text;
    float mean_conf = 0, min_char_conf = 0;
    float bbox[4] = {0, 0, 0, 0};
    uint64_t seq = 0;
};

struct FrameMeta {
    uint32_t stream_index = 0;
    uint64_t seq = 0;
    double wall_ms = 0, t_mono_s = 0;
    LetterboxGeom geom{};
    const std::vector<Stage2Read>* reads = nullptr;  // C++-internal; not part of the plugin ABI
};

struct AnalyzerEvent {
    std::string type;
    uint32_t track_id = 0;
    std::string fields_json;  // text of a JSON object
};

class Analyzer {
public:
    virtual ~Analyzer() = default;

    virtual const char* name() const = 0;
    // Number of f32 attributes emitted per track per frame.
    virtual uint32_t attr_count() const { return 0; }
    // false: may be enabled even when tracker.enabled=false (slot_coverage).
    virtual bool needs_tracks() const { return true; }
    // Minimum Caps.keypoints the model must provide (pose_angle, M1.17);
    // runtime add() rejects the analyzer when caps.keypoints < min_keypoints().
    virtual uint32_t min_keypoints() const { return 0; }
    // Runtime re-configure; on failure the old config and state are kept.
    virtual bool configure(const std::string& json, std::string& err) = 0;
    virtual void on_frame(const FrameMeta& m, const std::vector<Track>& tracks,
                          float* attrs /* tracks.size() * attr_count */,
                          std::vector<AnalyzerEvent>& out) = 0;
    // Called when a track is removed; may emit events (e.g. text_vote flush);
    // event seq/wall_ms are those of the frame the removal happened on.
    virtual void on_track_removed(uint32_t track_id, double t_mono_s,
                                  std::vector<AnalyzerEvent>& out) = 0;
};

// Built-in analyzer factory ("line_cross", "zone", ...). Returns nullptr
// with err set for unknown names.
std::unique_ptr<Analyzer> create_analyzer(const std::string& name, std::string& err);

// Load an application analyzer plugin (.so exporting vb_analyzer_entry).
// Fails when the library or symbol is missing or the ABI version differs
// (never silently skipped). The returned analyzer is unconfigured; call
// configure() with the plugin's JSON config.
std::unique_ptr<Analyzer> load_plugin_analyzer(const std::string& so_path, std::string& err);

}  // namespace vb
