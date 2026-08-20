#ifndef _ZONE_METRICS_H_
#define _ZONE_METRICS_H_

#include <deque>
#include <vector>
#include <algorithm>

#include "person_tracker.h"

namespace retail_vision {

struct ZoneSnapshot {
    int occupancy_count = 0;
    int browsing_count = 0;
    int engaged_count = 0;
    int assist_count = 0;
    int peak_customer = 0;
    float avg_dwell_time = 0.0f;
    float avg_engagement_time = 0.0f;
    float avg_velocity = 0.0f;
    int entry_count = 0;
    int exit_count = 0;
};

class ZoneMetrics {
public:
    ZoneMetrics();
    ~ZoneMetrics() = default;

    void setWindowDuration(float seconds);

    // Called every frame with current tracked persons
    void update(const StateCount& counts, int entry_count, int exit_count, float current_time_sec);

    // Called when a track is removed (via callback from PersonTracker)
    void onTrackRemoved(const TrackRecord& record);

    ZoneSnapshot getSnapshot() const;

private:
    void pruneWindow(float current_time);

    float window_duration_ = 60.0f;

    // Occupancy samples (1 per second) for peak calculation
    struct OccupancySample {
        float time;
        int count;
    };
    std::deque<OccupancySample> occupancy_samples_;
    float last_sample_time_ = 0.0f;

    // Removed track records within the window
    std::deque<TrackRecord> removed_tracks_;

    // Running sums for removed tracks in window
    float sum_dwell_time_ = 0.0f;
    float sum_engagement_time_ = 0.0f;
    float sum_avg_speed_ = 0.0f;
    int removed_count_ = 0;

    // UA: median filter for occupancy smoothing (window=5)
    static constexpr int SMOOTHING_WINDOW = 5;
    std::deque<int> occupancy_history_;
    int smoothOccupancy(int raw_count);

    // Latest state counts (updated every frame)
    StateCount current_counts_;
    int smoothed_occupancy_ = 0;
    int current_entry_count_ = 0;
    int current_exit_count_ = 0;
};

}  // namespace retail_vision

#endif  // _ZONE_METRICS_H_
