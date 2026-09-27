// Analyzer plugin loader + built-in analyzer factory (spec BASE-1 §6.2, M1.6).
#include <dlfcn.h>

#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "vb/analyzer.h"
#include "vb/runtime.h"  // load_plugin_api / validate_plugin_api / make_plugin_analyzer
#include "vb/vb_analyzer_abi.h"

#include "count_threshold.h"
#include "direction.h"
#include "dwell.h"
#include "line_cross.h"
#include "pose_angle.h"
#include "speed.h"
#include "zone.h"

namespace vb {
namespace {

struct EmitCtx {
    std::vector<AnalyzerEvent>* out;
};

void emit_cb(void* sink, const char* type, uint32_t track_id, const char* fields_json) {
    auto* out = static_cast<std::vector<AnalyzerEvent>*>(sink);
    AnalyzerEvent ev;
    ev.type = type ? type : "";
    ev.track_id = track_id;
    ev.fields_json = fields_json ? fields_json : "{}";
    out->push_back(std::move(ev));
}

class PluginAnalyzer : public Analyzer {
public:
    // `handle` owns the dlopen reference: the library stays loaded while any
    // instance (or the runtime's plugin registry) still holds a copy.
    PluginAnalyzer(std::shared_ptr<void> handle, const vb_analyzer_api* api)
        : handle_(std::move(handle)), api_(api) {}
    ~PluginAnalyzer() override {
        if (self_) api_->destroy(self_);
    }

    const char* name() const override { return api_->name ? api_->name : "plugin"; }
    uint32_t attr_count() const override { return api_->attr_count; }

    bool configure(const std::string& json, std::string& err) override {
        char ebuf[256] = {0};
        void* inst = api_->create(json.c_str(), ebuf, sizeof(ebuf));
        if (!inst) {
            err = ebuf[0] ? std::string(ebuf) : "plugin create failed";
            return false;  // old instance (if any) is kept
        }
        if (self_) api_->destroy(self_);
        self_ = inst;
        return true;
    }

    void on_frame(const FrameMeta& m, const std::vector<Track>& tracks, float* attrs,
                  std::vector<AnalyzerEvent>& out) override {
        if (!self_) return;
        std::vector<vb_track> ct(tracks.size());
        std::vector<float> kpts;
        size_t total = 0;
        for (const auto& t : tracks) total += t.kpts.size();
        kpts.reserve(total * 3);
        for (size_t i = 0; i < tracks.size(); ++i) {
            const Track& t = tracks[i];
            ct[i].track_id = t.track_id;
            ct[i].cx = t.det.cx;
            ct[i].cy = t.det.cy;
            ct[i].w = t.det.w;
            ct[i].h = t.det.h;
            ct[i].score = t.det.score;
            ct[i].class_id = t.det.class_id;
            ct[i].vx = t.vx;
            ct[i].vy = t.vy;
            ct[i].hits = t.hits;
            ct[i].misses = t.misses;
            ct[i].kpt_count = static_cast<uint32_t>(t.kpts.size());
            ct[i].kpts = nullptr;
            if (!t.kpts.empty()) {
                ct[i].kpts = kpts.data() + kpts.size();
                for (const auto& k : t.kpts) {
                    kpts.push_back(k.x);
                    kpts.push_back(k.y);
                    kpts.push_back(k.conf);
                }
            }
        }
        vb_frame_meta fm{};
        fm.stream_index = m.stream_index;
        fm.seq = m.seq;
        fm.wall_ms = m.wall_ms;
        fm.t_mono_s = m.t_mono_s;
        fm.src_w = m.geom.src_w;
        fm.src_h = m.geom.src_h;
        fm.model_w = m.geom.model_w;
        fm.model_h = m.geom.model_h;
        fm.scale = m.geom.scale;
        fm.pad_x = m.geom.pad_x;
        fm.pad_y = m.geom.pad_y;
        fm.align = static_cast<uint8_t>(m.geom.align);
        api_->on_frame(self_, &fm, ct.empty() ? nullptr : ct.data(), ct.size(), attrs,
                       &emit_cb, &out);
    }

    void on_track_removed(uint32_t track_id, double t_mono_s,
                          std::vector<AnalyzerEvent>& out) override {
        (void)out;
        if (!self_) return;
        api_->on_track_removed(self_, track_id, t_mono_s);
    }

private:
    std::shared_ptr<void> handle_;
    const vb_analyzer_api* api_ = nullptr;
    void* self_ = nullptr;
};

}  // namespace

bool validate_plugin_api(const vb_analyzer_api* api, std::string& err) {
    // api->abi was checked before this call; everything else in the table is
    // validated here so a plugin that leaves a mandatory callback null is
    // rejected at load time instead of crashing the first frame it runs on.
    if (!api->create) {
        err = "plugin api: create is null";
        return false;
    }
    if (!api->destroy) {
        err = "plugin api: destroy is null";
        return false;
    }
    if (!api->on_frame) {
        err = "plugin api: on_frame is null";
        return false;
    }
    if (!api->on_track_removed) {
        err = "plugin api: on_track_removed is null";
        return false;
    }
    if (!api->name || api->name[0] == '\0') {
        err = "plugin api: name is empty";
        return false;
    }
    if (api->attr_count > 0 && !api->attr_names) {
        err = "plugin api: attr_names is null with attr_count " +
              std::to_string(api->attr_count);
        return false;
    }
    // The declared attributes have to fit VBR1's attr_per_det (u8) and the
    // hello attribute list; a plugin asking for more can never be encoded.
    // (attr_names is a bare pointer in the ABI, so its length cannot be
    // checked from here without reading past the end of the array.)
    if (api->attr_count > 255) {
        err = "plugin api: attr_count " + std::to_string(api->attr_count) +
              " exceeds the 255 cap of VBR1";
        return false;
    }
    return true;
}

std::shared_ptr<void> load_plugin_api(const std::string& so_path,
                                      const vb_analyzer_api** api,
                                      std::string& err) {
    void* handle = dlopen(so_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        // A second dlerror() call may return nullptr (the error is consumed by
        // the first read), which would build std::string(nullptr).
        const char* derr = dlerror();
        err = std::string("plugin load failed: ") + (derr ? derr : so_path.c_str());
        return nullptr;
    }
    dlerror();
    auto entry = reinterpret_cast<const vb_analyzer_api* (*)()>(
        dlsym(handle, "vb_analyzer_entry"));
    const char* derr = dlerror();
    if (!entry || derr) {
        err = std::string("plugin has no vb_analyzer_entry: ") + (derr ? derr : so_path.c_str());
        dlclose(handle);
        return nullptr;
    }
    const vb_analyzer_api* table = entry();
    if (!table || table->abi != VB_ANALYZER_ABI) {
        err = "plugin abi mismatch: expected " + std::to_string(VB_ANALYZER_ABI) + ", got " +
              (table ? std::to_string(table->abi) : std::string("null"));
        dlclose(handle);
        return nullptr;
    }
    std::string api_err;
    if (!validate_plugin_api(table, api_err)) {
        err = so_path + ": " + api_err;
        dlclose(handle);
        return nullptr;
    }
    *api = table;
    return std::shared_ptr<void>(handle, [](void* h) { dlclose(h); });
}

std::unique_ptr<Analyzer> make_plugin_analyzer(std::shared_ptr<void> handle,
                                               const vb_analyzer_api* api) {
    return std::make_unique<PluginAnalyzer>(std::move(handle), api);
}

std::unique_ptr<Analyzer> load_plugin_analyzer(const std::string& so_path, std::string& err) {
    const vb_analyzer_api* api = nullptr;
    std::shared_ptr<void> handle = load_plugin_api(so_path, &api, err);
    if (!handle) return nullptr;
    return make_plugin_analyzer(std::move(handle), api);
}

std::unique_ptr<Analyzer> create_analyzer(const std::string& name, std::string& err) {
    if (name == "line_cross") return make_line_cross_analyzer();
    if (name == "zone") return make_zone_analyzer();
    if (name == "dwell") return make_dwell_analyzer();
    if (name == "speed") return make_speed_analyzer();
    if (name == "direction") return make_direction_analyzer();
    if (name == "count_threshold") return make_count_threshold_analyzer();
    if (name == "pose_angle") return make_pose_angle_analyzer();
    err = "unknown analyzer: " + name;
    return nullptr;
}

}  // namespace vb
