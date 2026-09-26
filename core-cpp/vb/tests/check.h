// Minimal test helper (spec BASE-1 §8 M1.3: CHECK macro, no gtest).
#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#define CHECK(expr)                                                          \
    do {                                                                     \
        if (!(expr)) {                                                       \
            std::fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__,       \
                         __LINE__, #expr);                                   \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                \
    do {                                                                     \
        double _va = (a), _vb = (b), _t = (tol);                             \
        if (!(std::fabs(_va - _vb) <= _t)) {                                 \
            std::fprintf(stderr, "CHECK_NEAR failed %s:%d: %s=%g vs %s=%g "  \
                                 "(tol %g)\n", __FILE__, __LINE__, #a,       \
                         _va, #b, _vb, _t);                                  \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

#define CHECK_STREQ(a, b)                                                    \
    do {                                                                     \
        std::string _sa(a), _sb(b);                                          \
        if (_sa != _sb) {                                                    \
            std::fprintf(stderr, "CHECK_STREQ failed %s:%d: \"%s\" vs "      \
                                 "\"%s\"\n", __FILE__, __LINE__, _sa.c_str(),\
                         _sb.c_str());                                       \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

inline std::string read_file(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        std::exit(1);
    }
    std::string data;
    char buf[65536];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) data.append(buf, n);
    std::fclose(f);
    return data;
}
