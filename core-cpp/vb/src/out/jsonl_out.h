// JsonlOut: one JSON record per line on stdout, fflush per line
// (spec BASE-1 §6.10.1, M1.18). Logs never go here (stderr only).
#pragma once

#include <cstdio>
#include <string>

namespace vb {

class JsonlOut {
public:
    void write(const std::string& line) {
        std::fwrite(line.data(), 1, line.size(), stdout);
        std::fputc('\n', stdout);
        std::fflush(stdout);
    }
};

}  // namespace vb
