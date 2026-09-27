// JsonlOut: one JSON record per line on stdout, fflush per line
// (spec BASE-1 §6.10.1, M1.18). Logs never go here (stderr only).
#pragma once

#include <cstdio>
#include <mutex>
#include <string>

namespace vb {

class JsonlOut {
public:
    // A record is one JSON line, so the body and its newline must be written as
    // one indivisible unit: the Writer thread emits events/frames while the
    // main thread emits status, and two unserialised writers interleave into
    // "JSON_A JSON_B\n\n" (one line carrying two documents, one empty line).
    void write(const std::string& line) {
        std::lock_guard<std::mutex> lk(mu_);
        std::fwrite(line.data(), 1, line.size(), stdout);
        std::fputc('\n', stdout);
        std::fflush(stdout);
    }

private:
    std::mutex mu_;
};

}  // namespace vb
