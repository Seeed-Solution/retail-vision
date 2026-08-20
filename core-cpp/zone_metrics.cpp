#include "zone_metrics.h"

#include <algorithm>

namespace retail_vision {

ZoneMetrics::ZoneMetrics() {}

void ZoneMetrics::setWindowDuration(float seconds) {
    window_duration_ = seconds;
}

void ZoneMetrics::pruneWindow(float current_time) {
    float cutoff = current_time - window_duration_;

    // Prune occupancy samples
    while (!occupancy_samples_.empty() && occupancy_samples_.front().time < cutoff) {
        occupancy_samples_.pop_front();
    }

    // Prune removed tracks and update running sums
    while (!removed_tracks_.empty() && removed_tracks_.front().removal_time < cutoff) {
        const auto& old = removed_tracks_.front();
        sum_dwell_time_ -= old.dwell_time;
        sum_engagement_time_ -= old.engagement_time;
        sum_avg_speed_ -= old.avg_speed;
        removed_count_--;
        removed_tracks_.pop_front();
    }
}

// UA: median filter over last 5 frames for occupancy smoothing
int ZoneMetrics::smoothOccupancy(int raw_count) {
    occupancy_history_.push_back(raw_count);
    while (occupancy_history_.size() > SMOOTHING_WINDOW) {
        occupancy_history_.pop_front();
    }

    std::vector<int> sorted(occupancy_history_.begin(), occupancy_history_.end());
    std::sort(sorted.begin(), sorted.end());
    return sorted[sorted.size() / 2];
}

void ZoneMetrics::update(const StateCount& counts, int entry_count, int exit_count, float current_time_sec) {
    current_counts_ = counts;
    current_entry_count_ = entry_count;
    current_exit_count_ = exit_count;

    // Smooth occupancy (UA median filter)
    smoothed_occupancy_ = smoothOccupancy(counts.total);

    // Sample occupancy once per second
    if (current_time_sec - last_sample_time_ >= 1.0f) {
        occupancy_samples_.push_back({current_time_sec, smoothed_occupancy_});
        last_sample_time_ = current_time_sec;
    }

    pruneWindow(current_time_sec);
}

void ZoneMetrics::onTrackRemoved(const TrackRecord& record) {
    removed_tracks_.push_back(record);
    sum_dwell_time_ += record.dwell_time;
    sum_engagement_time_ += record.engagement_time;
    sum_avg_speed_ += record.avg_speed;
    removed_count_++;
}

ZoneSnapshot ZoneMetrics::getSnapshot() const {
    ZoneSnapshot snap;

    snap.occupancy_count = smoothed_occupancy_;
    snap.browsing_count = current_counts_.browsing;
    snap.engaged_count = current_counts_.engaged;
    snap.assist_count = current_counts_.assistance;
    snap.entry_count = current_entry_count_;
    snap.exit_count = current_exit_count_;

    // Peak from occupancy samples
    snap.peak_customer = 0;
    for (const auto& s : occupancy_samples_) {
        if (s.count > snap.peak_customer) {
            snap.peak_customer = s.count;
        }
    }

    // Averages from removed tracks in window
    if (removed_count_ > 0) {
        snap.avg_dwell_time = sum_dwell_time_ / removed_count_;
        snap.avg_engagement_time = sum_engagement_time_ / removed_count_;
        snap.avg_velocity = sum_avg_speed_ / removed_count_;
    }

    return snap;
}

}  // namespace retail_vision
