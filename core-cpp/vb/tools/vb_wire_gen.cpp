// tools/vb_wire_gen.cpp: regenerate contracts/fixtures/vb/wire_case{1,2,3}.bin
// from the .json fixtures (spec BASE-1 §8 M1.4).
//
// Usage: vb_wire_gen <fixture.json> <out.bin> [...]
#include <cstdio>
#include <fstream>

#include "../tests/check.h"
#include "../tests/wire_case_util.h"

int main(int argc, char** argv) {
    if (argc < 3 || (argc - 1) % 2 != 0) {
        std::fprintf(stderr, "usage: vb_wire_gen <case.json> <out.bin> [<case.json> <out.bin> ...]\n");
        return 2;
    }
    for (int i = 1; i + 1 < argc; i += 2) {
        auto j = vb::json_parse(read_file(argv[i]));
        std::vector<uint8_t> enc;
        const std::string magic = j.at("magic").get<std::string>();
        if (magic == "VBR1") {
            auto rec = vb::wire_frame_from_case_json(j);
            vb::wire_encode_vbr1(rec, enc);
        } else if (magic == "VBT1") {
            auto tf = vb::dev_tensor_frame_from_case_json(j);
            std::string err;
            CHECK(vb::wire_encode_vbt1(tf, enc, err));
        } else {
            CHECK(!"unsupported fixture magic");
        }
        std::ofstream out(argv[i + 1], std::ios::binary);
        out.write(reinterpret_cast<const char*>(enc.data()),
                  static_cast<std::streamsize>(enc.size()));
        out.close();
        CHECK(out.good());
        std::printf("%s -> %s (%zu bytes)\n", argv[i], argv[i + 1], enc.size());
    }
    return 0;
}
