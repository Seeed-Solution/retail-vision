#ifndef _RETAIL_DETECTION_H_
#define _RETAIL_DETECTION_H_

// Standalone replacement for the SG200X solution's detector.h.
// The tracker only ever consumed DetectionBox from it; the sscma Detector class
// has no counterpart here because detection happens inside hailonet.

namespace retail_vision {

struct DetectionBox {
    float x;      // Normalized center x [0-1] in model input space
    float y;      // Normalized center y [0-1] in model input space
    float w;      // Normalized width [0-1]
    float h;      // Normalized height [0-1]
    float score;  // Detection confidence [0-1]
    int target;   // COCO class id; the tracker keeps target == 0 (person)
};

}  // namespace retail_vision

#endif  // _RETAIL_DETECTION_H_
