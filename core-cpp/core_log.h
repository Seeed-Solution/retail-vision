#ifndef _RETAIL_CORE_LOG_H_
#define _RETAIL_CORE_LOG_H_

// person_tracker.cpp logs through MA_LOGI, which on the SG2002 comes from
// <sscma.h>. That header is the tracker's only tie to the sscma SDK — no ma::
// symbol appears anywhere in the file — so this shim replaces the include and
// leaves every call site untouched. On the SG2002 build, define
// RETAIL_CORE_USE_SSCMA and the calls go back to the SDK's logger; anywhere
// else they go to stderr, gated so the publish loop stays quiet by default.

#ifdef RETAIL_CORE_USE_SSCMA

#include <sscma.h>

#else

#include <cstdio>
#include <cstdlib>

#ifndef MA_LOGI
#define MA_LOGI(tag, ...)                        \
    do {                                         \
        if (std::getenv("RV_VERBOSE")) {         \
            std::fprintf(stderr, "[%s] ", tag);  \
            std::fprintf(stderr, __VA_ARGS__);   \
            std::fprintf(stderr, "\n");          \
        }                                        \
    } while (0)
#endif

#endif  // RETAIL_CORE_USE_SSCMA

#endif  // _RETAIL_CORE_LOG_H_
