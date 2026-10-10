// M1.4 wire-format test: C++ encode of wire_case*.json must be byte-identical
// to wire_case*.bin (spec BASE-1 §8 M1.4).
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "check.h"
#include "vb/wire.h"
#include "vb/json.h"
#include "wire_case_util.h"

namespace {

std::vector<uint8_t> read_bin(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        std::exit(1);
    }
    std::vector<uint8_t> data;
    uint8_t buf[65536];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) data.insert(data.end(), buf, buf + n);
    std::fclose(f);
    return data;
}

void check_case(const std::string& base) {
    auto j = vb::json_parse(read_file(base + ".json"));
    CHECK_STREQ(j.at("magic").get<std::string>(), "VBR1");
    auto rec = vb::wire_frame_from_case_json(j);
    std::vector<uint8_t> enc;
    vb::wire_encode_vbr1(rec, enc);
    auto expect = read_bin(base + ".bin");
    CHECK(enc.size() == expect.size());
    if (enc != expect) {
        for (size_t i = 0; i < enc.size(); ++i) {
            if (enc[i] != expect[i]) {
                std::fprintf(stderr, "%s: first diff at byte %zu: got %02x want %02x\n",
                             base.c_str(), i, enc[i], expect[i]);
                std::exit(1);
            }
        }
    }
    std::printf("%s: %zu bytes identical\n", base.c_str(), enc.size());
}

}  // namespace

int main(int argc, char** argv) {
    const std::string fixtures = (argc > 1) ? argv[1] : "contracts/fixtures/vb";
    check_case(fixtures + "/wire_case1");
    check_case(fixtures + "/wire_case2");
    check_case(fixtures + "/wire_case3");

    // VBE1 / VBC1 framing: 8-byte header + verbatim UTF-8 JSON body.
    {
        const std::string body = R"({"op":"stats","rss_kb":123456})";
        std::vector<uint8_t> enc;
        vb::wire_encode_json_record("VBC1", body, enc);
        CHECK(enc.size() == 8 + body.size());
        CHECK(std::memcmp(enc.data(), "VBC1", 4) == 0);
        uint32_t len = enc[4] | (enc[5] << 8) | (enc[6] << 16) | (uint32_t(enc[7]) << 24);
        CHECK(len == body.size());
        CHECK(std::memcmp(enc.data() + 8, body.data(), body.size()) == 0);

        vb::wire_encode_json_record("VBE1", body, enc);
        CHECK(enc.size() == 2 * (8 + body.size()));
    }

    // VBS1: u32 json_len + json + payload.
    {
        const std::string meta = R"({"req":"r-9","w":8,"h":6})";
        const uint8_t payload[] = {0xFF, 0xD8, 0xFF, 0xE0, 0x00};
        std::vector<uint8_t> enc;
        vb::wire_encode_vbs1(meta, payload, sizeof(payload), enc);
        CHECK(enc.size() == 8 + 4 + meta.size() + sizeof(payload));
        CHECK(std::memcmp(enc.data(), "VBS1", 4) == 0);
        uint32_t body_len = enc[4] | (enc[5] << 8) | (enc[6] << 16) | (uint32_t(enc[7]) << 24);
        CHECK(body_len == 4 + meta.size() + sizeof(payload));
        uint32_t json_len = enc[8] | (enc[9] << 8) | (enc[10] << 16) | (uint32_t(enc[11]) << 24);
        CHECK(json_len == meta.size());
        CHECK(enc.back() == 0x00);
    }

    // VBR1 header sanity on a tiny record.
    {
        vb::WireFrameRec r;
        r.stream_index = 9;
        r.dets.push_back(vb::WireDet{0.5f, 0.5f, 0.2f, 0.2f, 0.9f, 1, 7});
        r.attr_per_det = 1;
        r.attrs = {0.25f};
        std::vector<uint8_t> enc;
        vb::wire_encode_vbr1(r, enc);
        CHECK(enc.size() == 8 + 64 + 28 + 4);  // header + fixed body + 1 det + 1 attr
        CHECK(enc[58] == 1 && enc[59] == 0);  // n_det little-endian
        CHECK(enc[68] == 1);                  // attr_per_det
        CHECK(enc[69] == 0);                  // reserved
        CHECK(enc[70] == 0 && enc[71] == 0);  // reserved
    }

    // B9: a record whose declared counts disagree with its payload must fail
    // the encode. Emitting it would make body_len describe fewer bytes than
    // the payload writes, and the leftovers would be read as the next record's
    // magic (measured: a body declared 1112 bytes and wrote 1116).
    {
        vb::WireFrameRec r;
        r.dets.resize(0x10000);  // beyond the u16 n_det field
        std::vector<uint8_t> enc;
        std::string err;
        CHECK(!vb::wire_encode_vbr1(r, enc, &err));
        CHECK(!err.empty());
        CHECK(enc.empty());
    }
    {
        // One detection, kpt_per_det = 17, but a single keypoint in the vector.
        vb::WireFrameRec r;
        r.dets.push_back(vb::WireDet{0.5f, 0.5f, 0.2f, 0.2f, 0.9f, 1, 7});
        r.kpt_per_det = 17;
        r.kpts = {0.1f, 0.2f, 0.9f};
        std::vector<uint8_t> enc;
        std::string err;
        CHECK(!vb::wire_encode_vbr1(r, enc, &err));
        CHECK(enc.empty());
    }
    {
        // The reported shape: attr_per_det says 2, the payload carries 1.
        vb::WireFrameRec r;
        r.dets.push_back(vb::WireDet{0.5f, 0.5f, 0.2f, 0.2f, 0.9f, 1, 7});
        r.attr_per_det = 2;
        r.attrs = {0.25f};
        std::vector<uint8_t> enc;
        std::string err;
        CHECK(!vb::wire_encode_vbr1(r, enc, &err));
        CHECK(enc.empty());
    }
    {
        // Matching counts still encode, through both call forms (the two
        // argument form stays source-compatible with existing callers).
        vb::WireFrameRec r;
        r.dets.push_back(vb::WireDet{0.5f, 0.5f, 0.2f, 0.2f, 0.9f, 1, 7});
        r.kpt_per_det = 1;
        r.kpts = {0.1f, 0.2f, 0.9f};
        r.attr_per_det = 1;
        r.attrs = {0.5f};
        std::vector<uint8_t> enc;
        vb::wire_encode_vbr1(r, enc);  // 2-arg form: return value ignored
        CHECK(enc.size() == 8 + 64 + 28 + 12 + 4);
        std::vector<uint8_t> enc2;
        std::string err;
        CHECK(vb::wire_encode_vbr1(r, enc2, &err));
        CHECK(err.empty());
        CHECK(enc2 == enc);
    }

    std::printf("wire: all checks passed\n");
    return 0;
}
