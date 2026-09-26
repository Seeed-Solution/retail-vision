// vb.event/1 / vb.frame/1 JSON construction (spec BASE-1 §6.10.2, M1.18).
#pragma once

#include <string>

#include "vb/json.h"
#include "vb/wire.h"

namespace vb {

// vb.event/1 from a decoded VBE1 body (stream_index/seq/wall_ms/analyzer/
// type/track_id/fields); wall_ms is truncated to an integer ts_ms.
Json event_json(const std::string& device_id, const std::string& stream_id,
                const Json& vbe1_body);

// vb.frame/1 from a VBR1 record; box/keypoints are letterbox-inverse mapped
// to source-normalized coords. keypoints/attrs keys are omitted when empty.
Json frame_json(const std::string& device_id, const std::string& stream_id,
                const WireFrameRec& r);

}  // namespace vb
