#include "mqtt_payload.h"

#include <algorithm>
#include <iomanip>
#include <sstream>

namespace retail_vision {

std::string buildVisionJson(uint64_t timestamp_ms, uint32_t frame_id,
                            float fps, float inference_time_ms,
                            const ZoneSnapshot& zone,
                            const std::vector<TrackedPerson>& persons,
                            int frame_width, int frame_height,
                            int model_width, int model_height) {
    std::ostringstream j;
    j << std::fixed;

    float fw = static_cast<float>(frame_width);
    float fh = static_cast<float>(frame_height);
    float mw = static_cast<float>(model_width);
    float mh = static_cast<float>(model_height);

    // Letterbox correction: VPSS fits the source into model input preserving aspect ratio.
    // For 16:9 display → square model: width fills, height has padding.
    // Compute the content area within the model's input space.
    float display_aspect = fw / fh;
    float model_aspect = mw / mh;

    // Scale factor and padding in normalized [0,1] model space
    float scale_x = 1.0f, scale_y = 1.0f;
    float offset_x = 0.0f, offset_y = 0.0f;

    if (display_aspect > model_aspect) {
        // Landscape: width fills model, height is padded
        float content_h = mh * (model_aspect / display_aspect);  // e.g. 640*(1.0/1.778)=360
        scale_y = content_h / mh;       // 360/640 = 0.5625
        offset_y = (1.0f - scale_y) / 2.0f;  // (1-0.5625)/2 = 0.21875
    } else if (display_aspect < model_aspect) {
        // Portrait: height fills model, width is padded
        float content_w = mw * (display_aspect / model_aspect);
        scale_x = content_w / mw;
        offset_x = (1.0f - scale_x) / 2.0f;
    }

    j << "{";
    j << "\"timestamp\":" << timestamp_ms;
    j << ",\"frame_id\":" << frame_id;
    j << ",\"frame_width\":" << frame_width;
    j << ",\"frame_height\":" << frame_height;
    j << ",\"fps\":" << std::setprecision(1) << fps;
    j << ",\"inference_time_ms\":" << std::setprecision(1) << inference_time_ms;

    // Zone metrics
    j << ",\"zone\":{";
    j << "\"occupancy_count\":" << zone.occupancy_count;
    j << ",\"browsing_count\":" << zone.browsing_count;
    j << ",\"engaged_count\":" << zone.engaged_count;
    j << ",\"assist_count\":" << zone.assist_count;
    j << ",\"peak_customer\":" << zone.peak_customer;
    j << ",\"avg_dwell_time\":" << std::setprecision(1) << zone.avg_dwell_time;
    j << ",\"avg_engagement_time\":" << std::setprecision(1) << zone.avg_engagement_time;
    j << ",\"avg_velocity\":" << std::setprecision(2) << zone.avg_velocity;
    j << ",\"entry_count\":" << zone.entry_count;
    j << ",\"exit_count\":" << zone.exit_count;
    j << "}";

    // Persons array — display-normalized coords (top-left x,y + w,h in [0,1]), matching draw script format
    j << ",\"persons\":[";
    for (size_t i = 0; i < persons.size(); ++i) {
        const auto& p = persons[i];
        if (i > 0) j << ",";

        // Undo letterbox: convert from model-normalized [0,1] to display-normalized [0,1]
        float real_cx = (p.detection.x - offset_x) / scale_x;
        float real_cy = (p.detection.y - offset_y) / scale_y;
        float real_w  = p.detection.w / scale_x;
        float real_h  = p.detection.h / scale_y;

        // Top-left normalized coords, intersected with the frame. Undoing the
        // letterbox can push a box past the frame edge when the detector
        // predicts into the padding band: a person cannot be taller than the
        // picture, so the part outside is not a real detection. Without this
        // the published bbox left its documented [0,1] range (observed
        // y=-0.05, h=1.10) and overlay consumers drew off-frame.
        float x0 = std::max(0.0f, real_cx - real_w / 2.0f);
        float y0 = std::max(0.0f, real_cy - real_h / 2.0f);
        float x1 = std::min(1.0f, real_cx + real_w / 2.0f);
        float y1 = std::min(1.0f, real_cy + real_h / 2.0f);
        float bx = x0;
        float by = y0;
        real_w = std::max(0.0f, x1 - x0);
        real_h = std::max(0.0f, y1 - y0);

        j << "{";
        // Index within this batch. Every person in one message shares a
        // timestamp, so a consumer storing them as time series needs a
        // distinguishing key; index rather than track_id keeps that key
        // bounded by people-per-frame instead of growing for the life of
        // the deployment.
        j << "\"slot\":" << i;
        j << ",\"track_id\":" << p.track_id;
        j << ",\"confidence\":" << std::setprecision(2) << p.detection.score;

        // Centre as a percentage of the frame. bbox below is normalized
        // [0,1] top-left+size; this is the same point expressed the way
        // dashboards and floor-plan calibration consume it, so neither has
        // to know the sensor resolution.
        j << ",\"cx_pct\":" << std::setprecision(1) << (real_cx * 100.0f);
        j << ",\"cy_pct\":" << std::setprecision(1) << (real_cy * 100.0f);

        j << ",\"bbox\":{";
        j << "\"x\":" << std::setprecision(4) << bx;
        j << ",\"y\":" << by;
        j << ",\"w\":" << real_w;
        j << ",\"h\":" << real_h;
        j << "}";

        j << ",\"velocity\":{";
        j << "\"vx\":" << std::setprecision(2) << (p.velocity_x / fw);
        j << ",\"vy\":" << std::setprecision(2) << (p.velocity_y / fh);
        j << ",\"speed_m_s\":" << std::setprecision(2) << p.speed_m_s;
        j << "}";

        j << ",\"state\":\"" << getDwellStateName(p.dwell_state) << "\"";
        j << ",\"dwell_duration\":" << std::setprecision(1) << p.dwell_duration_sec;

        j << "}";
    }
    j << "]";

    j << "}";

    return j.str();
}

}  // namespace retail_vision
