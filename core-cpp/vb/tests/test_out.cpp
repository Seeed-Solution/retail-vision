// Output-layer tests (spec BASE-1 §6.10.2 / §6.10.3, M1.18/M1.19).
//
//   - §6.10.2 rules 1-4 of src/out/event_json.cpp: recursive field rounding,
//     half-to-even timestamps, double intermediates and [0, 1] clipping.
//   - MQTT fixed-header framing (src/out/mqtt_frame.h), incl. the malformed
//     Remaining Length that used to make the parse position wrap.
//   - CONNECT flag/payload agreement (src/out/mqtt_lite.cpp) over a loopback
//     socket: the flags must describe the credentials actually written.
#include <chrono>
#include <cstdint>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include "check.h"
#include "out/event_json.h"
#include "out/mqtt_frame.h"
#include "out/jsonl_out.h"
#include "out/mqtt_lite.h"
#include "vb/json.h"
#include "vb/types.h"
#include "vb/wire.h"

using vb::Json;

namespace {

double now_s() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}

// Mirror of event_json.cpp's num6(): %.6f then parse back.
double num6(double v) {
    char buf[352];
    std::snprintf(buf, sizeof buf, "%.6f", v);
    return std::strtod(buf, nullptr);
}

vb::WireDet det(float cx, float cy, float w, float h) {
    vb::WireDet d;
    d.cx = cx;
    d.cy = cy;
    d.w = w;
    d.h = h;
    d.score = 0.5f;
    return d;
}

// 1280x720 letterboxed into 640x640 (scale 0.5, pad_y 140): the geometry that
// produces out-of-frame coordinates for full-canvas model detections.
vb::WireFrameRec geom_frame() {
    vb::WireFrameRec r;
    r.stream_index = 0;
    r.seq = 120;
    r.wall_ms = 1790000000000.0;
    r.src_w = 1280;
    r.src_h = 720;
    r.model_w = 640;
    r.model_h = 640;
    const vb::LetterboxGeom g =
        vb::LetterboxGeom::fit(1280, 720, 640, 640, vb::Align::Center);
    r.scale = g.scale;
    r.pad_x = g.pad_x;
    r.pad_y = g.pad_y;
    r.align = 0;
    r.inference_ms = 12.4f;
    return r;
}

// ---- §6.10.2 rule 1: fields are rounded recursively ----

void test_event_fields_recursive_rounding() {
    Json body = Json::object();
    body["seq"] = 120;
    body["wall_ms"] = 1790000000000.0;
    body["analyzer"] = "line_cross";
    body["type"] = "line_cross";
    body["track_id"] = 7;
    Json fields = Json::object();
    fields["line_id"] = "door";
    fields["direction"] = "forward";
    Json anchor = Json::array();
    anchor.push_back(double(0.52f));  // 0.51999998092651367 as a double
    anchor.push_back(double(0.61f));
    fields["anchor"] = anchor;
    fields["class_id"] = 0;           // integer
    fields["count"] = 3;              // integer
    fields["active"] = true;          // boolean
    fields["score"] = double(0.81f);  // 0.81000000238418579 as a double
    Json nested = Json::object();
    nested["ratio"] = double(1.0f / 3.0f);  // 0.3333333432674408
    Json ids = Json::array();
    ids.push_back(1);
    ids.push_back(2);
    ids.push_back(3);
    nested["ids"] = ids;
    nested["flag"] = false;
    Json item = Json::object();
    item["angle"] = double(2.5f);
    Json deep = Json::array();
    deep.push_back(item);
    nested["list"] = deep;
    fields["nested"] = nested;
    body["fields"] = fields;

    Json j = vb::event_json("door-01", "door", body);
    CHECK_STREQ(j.at("schema").get<std::string>(), "vb.event/1");
    CHECK_STREQ(j.at("device_id").get<std::string>(), "door-01");
    CHECK_STREQ(j.at("stream_id").get<std::string>(), "door");
    CHECK(j.at("ts_ms").get<int64_t>() == 1790000000000LL);

    const Json& f = j.at("fields");
    // Floats, nested or not, are rounded to 6 decimals.
    CHECK(f.at("score").is_number_float());
    CHECK(f.at("score").get<double>() == 0.81);
    CHECK(f.at("anchor").at(0).get<double>() == 0.52);
    CHECK(f.at("anchor").at(1).get<double>() == 0.61);
    CHECK(f.at("nested").at("ratio").get<double>() == 0.333333);
    CHECK(f.at("nested").at("list").at(0).at("angle").get<double>() == 2.5);
    // Non-floats keep their JSON type: rounding them would change the payload.
    CHECK(f.at("line_id").is_string());
    CHECK_STREQ(f.at("line_id").get<std::string>(), "door");
    CHECK(f.at("class_id").is_number_integer());
    CHECK(f.at("class_id").get<int>() == 0);
    CHECK(f.at("count").is_number_integer());
    CHECK(f.at("count").get<int>() == 3);
    CHECK(f.at("active").is_boolean());
    CHECK(f.at("active").get<bool>());
    CHECK(f.at("nested").at("ids").is_array());
    CHECK(f.at("nested").at("ids").at(0).is_number_integer());
    CHECK(f.at("nested").at("ids").at(2).get<int>() == 3);
    CHECK(f.at("nested").at("flag").is_boolean());
    CHECK(!f.at("nested").at("flag").get<bool>());

    // The unrounded float32 expansions must not reach the record.
    const std::string out = vb::json_dump(j);
    CHECK(out.find("0.8100000023841858") == std::string::npos);
    CHECK(out.find("0.5199999809265137") == std::string::npos);
    CHECK(out.find("0.3333333432674408") == std::string::npos);
    CHECK(out.find("\"score\":0.81") != std::string::npos);

    // The same rule applies to a body parsed from JSON text (the VBE1 path).
    Json parsed = vb::json_parse(
        "{\"seq\":1,\"wall_ms\":1000,\"analyzer\":\"zone\",\"type\":\"zone_enter\","
        "\"track_id\":2,\"fields\":{\"score\":0.8100000023841858,\"hops\":2,"
        "\"ok\":true,\"pts\":[0.30000001192092896,0.7000000476837158]}}");
    Json jp = vb::event_json("d", "s", parsed);
    CHECK(jp.at("fields").at("score").get<double>() == 0.81);
    CHECK(jp.at("fields").at("hops").is_number_integer());
    CHECK(jp.at("fields").at("ok").is_boolean());
    CHECK(jp.at("fields").at("pts").at(0).get<double>() == 0.3);
    CHECK(jp.at("fields").at("pts").at(1).get<double>() == 0.7);
}

// ---- §6.10.2 rule 4: ts_ms is half-to-even, not half-away-from-zero ----

void test_ts_ms_half_to_even() {
    struct Case {
        double wall_ms;
        int64_t expect;
    };
    const Case cases[] = {
        {1000.5, 1000},  // llround() gave 1001 here; Python round() gives 1000
        {1001.5, 1002},
        {0.5, 0},
        {1.5, 2},
        {2.5, 2},
        {-0.5, 0},
        {-1.5, -2},
        {-2.5, -2},
        {1000.4999, 1000},
        {1010.500001, 1011},
        {1790000000000.0, 1790000000000LL},
    };
    for (const Case& c : cases) {
        Json body = Json::object();
        body["wall_ms"] = c.wall_ms;
        Json j = vb::event_json("d", "s", body);
        if (j.at("ts_ms").get<int64_t>() != c.expect) {
            std::fprintf(stderr, "ts_ms: wall_ms=%g -> %lld, want %lld\n", c.wall_ms,
                         static_cast<long long>(j.at("ts_ms").get<int64_t>()),
                         static_cast<long long>(c.expect));
            std::exit(1);
        }
    }
    // The frame path rounds the same way (it used to go through llround too).
    vb::WireFrameRec r = geom_frame();
    r.wall_ms = 1000.5;
    CHECK(vb::frame_json("d", "s", r).at("ts_ms").get<int64_t>() == 1000);
    r.wall_ms = 1001.5;
    CHECK(vb::frame_json("d", "s", r).at("ts_ms").get<int64_t>() == 1002);
}

// ---- §6.10.2 rule 2: the inverse transform runs in double ----

void test_box_inverse_uses_double() {
    const vb::LetterboxGeom g =
        vb::LetterboxGeom::fit(1280, 720, 640, 640, vb::Align::Center);
    const float w = 0.05f;  // keeps the box inside [0, 1] so clipping cannot
                            // mask the rounding difference
    // Sweep the float32 bit pattern between 0.10f and 0.90f (a decimal grid
    // would sit on multiples of 1e-6 and never straddle a rounding boundary).
    uint32_t bits = 0, end_bits = 0;
    const float lo = 0.10f, hi = 0.90f;
    std::memcpy(&bits, &lo, 4);
    std::memcpy(&end_bits, &hi, 4);
    float picked = 0.0f;
    double d_x0 = 0.0, f_x0 = 0.0;
    bool found = false;
    for (; bits < end_bits && !found; bits += 7) {
        float cx = 0.0f;
        std::memcpy(&cx, &bits, 4);
        // double intermediates: what the output layer must use.
        d_x0 = ((double(cx) * double(g.model_w) - double(g.pad_x)) /
                double(g.scale)) /
                   double(g.src_w) -
               0.5 * (double(w) * double(g.model_w) /
                      (double(g.scale) * double(g.src_w)));
        // float32 intermediates: the old path through box_to_source_norm().
        const float scx_f =
            ((cx * float(g.model_w) - g.pad_x) / g.scale) / float(g.src_w);
        const float sw_f = w * float(g.model_w) / (g.scale * float(g.src_w));
        f_x0 = double(scx_f - sw_f / 2.0f);
        if (num6(d_x0) != num6(f_x0)) {
            found = true;
            picked = cx;
        }
    }
    CHECK(found);  // the scan must find a discriminating input
    CHECK(num6(d_x0) != num6(f_x0));

    vb::WireFrameRec r = geom_frame();
    r.dets.push_back(det(picked, 0.5f, w, w));
    Json j = vb::frame_json("d", "s", r);
    const double x0 = j.at("detections").at(0).at("box").at(0).get<double>();
    CHECK(x0 > 0.0 && x0 < 1.0);  // not clipped, so the value is the transform
    CHECK(x0 == num6(d_x0));      // exactly the double-precision result
    CHECK(x0 != num6(f_x0));      // and visibly not the float32 one
}

// ---- §6.10.2 rule 3: box/keypoint coordinates are clipped to [0, 1] ----

void test_coordinates_clipped() {
    vb::WireFrameRec r = geom_frame();
    r.dets.push_back(det(0.5f, 0.5f, 1.0f, 1.0f));  // full model canvas
    r.dets.push_back(det(0.5f, 0.5f, 0.1f, 0.1f));  // carries the keypoints
    r.kpt_per_det = 3;
    for (int i = 0; i < 9; ++i) r.kpts.push_back(0.0f);  // first detection's
    // Model-canvas corners: x 0/1 maps to 0/1, y 0/1 maps outside the frame
    // because of the 140 px letterbox pad (y = -0.389 / 1.389).
    const float kp[9] = {0.0f, 0.0f, 0.9f, 1.0f, 1.0f, 0.9f, 0.5f, 0.5f, 1.5f};
    for (float v : kp) r.kpts.push_back(v);

    Json j = vb::frame_json("d", "s", r);
    const Json& box = j.at("detections").at(0).at("box");
    CHECK(box.size() == 4);
    CHECK(box.at(0).get<double>() == 0.0);
    CHECK(box.at(1).get<double>() == 0.0);
    CHECK(box.at(2).get<double>() == 1.0);
    CHECK(box.at(3).get<double>() == 1.0);
    CHECK(vb::json_dump(box).find("-") == std::string::npos);  // no -0.0 either

    const Json& kps = j.at("detections").at(1).at("keypoints");
    CHECK(kps.size() == 3);
    CHECK(kps.at(0).at(0).get<double>() == 0.0);
    CHECK(kps.at(0).at(1).get<double>() == 0.0);  // clipped, not -0.389
    CHECK(kps.at(0).at(2).get<double>() == 0.9);
    CHECK(kps.at(1).at(0).get<double>() == 1.0);
    CHECK(kps.at(1).at(1).get<double>() == 1.0);  // clipped, not 1.389
    CHECK(kps.at(2).at(0).get<double>() == 0.5);
    CHECK(kps.at(2).at(1).get<double>() == 0.5);
    // A confidence is not a coordinate: it is rounded, never clipped.
    CHECK(kps.at(2).at(2).get<double>() == 1.5);

    // keypoints/attrs keys are omitted when the record carries none.
    vb::WireFrameRec plain = geom_frame();
    plain.dets.push_back(det(0.5f, 0.5f, 0.1f, 0.1f));
    Json jp = vb::frame_json("d", "s", plain);
    CHECK(!jp.at("detections").at(0).contains("keypoints"));
    CHECK(!jp.at("detections").at(0).contains("attrs"));
    CHECK(jp.at("inference_ms").get<double>() == num6(double(12.4f)));
}

// ---- §6.10.3 MQTT fixed-header framing ----

void test_mqtt_framing() {
    uint8_t type = 0;
    std::string body;

    // One complete PUBACK (type 4, pid 0x0007).
    {
        std::string buf("\x40\x02\x00\x07", 4);
        CHECK(vb::mqtt_take_packet(buf, type, body) == vb::MqttFrame::Packet);
        CHECK(type == 4);
        CHECK(body.size() == 2);
        CHECK(uint8_t(body[0]) == 0x00 && uint8_t(body[1]) == 0x07);
        CHECK(buf.empty());
    }
    // A fragmented packet is not consumed until its last byte arrives: a
    // PUBACK split across TCP segments used to be discarded every round, so
    // the QoS1 message was re-sent after every 5 s timeout.
    {
        std::string buf("\x40\x02\x07", 3);
        CHECK(vb::mqtt_take_packet(buf, type, body) == vb::MqttFrame::NeedMore);
        CHECK(buf.size() == 3);
        buf.push_back('\x07');
        CHECK(vb::mqtt_take_packet(buf, type, body) == vb::MqttFrame::Packet);
        CHECK(uint8_t(body[0]) == 0x07 && uint8_t(body[1]) == 0x07);
        CHECK(buf.empty());
    }
    // The Remaining Length itself may be split across reads.
    {
        std::string buf("\x30", 1);
        CHECK(vb::mqtt_take_packet(buf, type, body) == vb::MqttFrame::NeedMore);
        buf.push_back('\x80');  // continuation bit: one more length byte
        CHECK(vb::mqtt_take_packet(buf, type, body) == vb::MqttFrame::NeedMore);
        buf.push_back('\x01');  // remaining length 128
        CHECK(vb::mqtt_take_packet(buf, type, body) == vb::MqttFrame::NeedMore);
        buf.append(128, 'x');
        CHECK(vb::mqtt_take_packet(buf, type, body) == vb::MqttFrame::Packet);
        CHECK(type == 3 && body.size() == 128 && buf.empty());
    }
    // The Remaining Length that decodes to 2^64-11 (ten length bytes instead of
    // four). It must be rejected as malformed, nothing consumed, instead of
    // wrapping the parse position back to 0 and spinning the loop forever.
    {
        std::string buf("\x30\xF5", 2);
        for (int i = 0; i < 8; ++i) buf.push_back('\xFF');
        buf.push_back('\x01');
        CHECK(vb::mqtt_take_packet(buf, type, body) == vb::MqttFrame::Malformed);
        CHECK(buf.size() == 11);  // caller drops the connection, buffer intact
    }
    // A 4th length byte that still has its continuation bit set is a 5-byte
    // encoding, which MQTT does not allow.
    {
        std::string buf("\x40\x80\x80\x80\x80\x01", 6);
        CHECK(vb::mqtt_take_packet(buf, type, body) == vb::MqttFrame::Malformed);
    }
    // Four length bytes are legal by themselves, but an announced packet this
    // client cannot receive (2097152 > 65536) is refused before buffering it.
    {
        std::string buf("\x30\x80\x80\x80\x01", 5);
        CHECK(vb::mqtt_take_packet(buf, type, body) == vb::MqttFrame::Malformed);
    }
    // Zero remaining length is legal (PINGRESP) and consumes exactly 2 bytes,
    // so the position still advances.
    {
        std::string buf("\xD0\x00", 2);
        CHECK(vb::mqtt_take_packet(buf, type, body) == vb::MqttFrame::Packet);
        CHECK(type == 13 && body.empty() && buf.empty());
    }
    // A valid packet in front of a malformed one is still parsed, and the
    // malformed one never returns Packet without consuming.
    {
        std::string buf;
        buf.append("\x40\x02\x00\x01", 4);
        buf.push_back('\x30');
        for (int i = 0; i < 10; ++i) buf.push_back('\xFF');
        CHECK(vb::mqtt_take_packet(buf, type, body) == vb::MqttFrame::Packet);
        CHECK(uint8_t(body[1]) == 0x01);
        CHECK(vb::mqtt_take_packet(buf, type, body) == vb::MqttFrame::Malformed);
    }
    // 1000 back-to-back packets parse in a bounded loop and drain completely.
    {
        std::string buf;
        for (int i = 0; i < 1000; ++i) buf.append("\x40\x02\x03\xE7", 4);
        int n = 0;
        while (vb::mqtt_take_packet(buf, type, body) == vb::MqttFrame::Packet) {
            CHECK(type == 4 && body.size() == 2);
            ++n;
            CHECK(n <= 1000);
        }
        CHECK(n == 1000);
        CHECK(buf.empty());
    }
}

// ---- §6.10.1 JsonlOut: one whole record per line under concurrent writers ----

void test_jsonl_records_not_interleaved() {
    // The Writer thread emits events/frames while the status timer emits
    // vb.status/1; a record split into body + newline used to interleave into
    // "JSON_A JSON_B\n\n" (one line with two documents, one empty line).
    char path[] = "/tmp/vb_jsonl_XXXXXX";
    const int fd = ::mkstemp(path);
    CHECK(fd >= 0);
    const int saved = ::dup(1);
    CHECK(saved >= 0);
    CHECK(::dup2(fd, 1) == 1);
    {
        vb::JsonlOut out;
        std::thread a([&out] {
            for (int i = 0; i < 2000; ++i)
                out.write(std::string(200, 'a') + std::to_string(i));
        });
        std::thread b([&out] {
            for (int i = 0; i < 2000; ++i)
                out.write(std::string(200, 'b') + std::to_string(i));
        });
        a.join();
        b.join();
    }
    std::fflush(stdout);
    CHECK(::dup2(saved, 1) == 1);
    ::close(saved);
    ::close(fd);
    const std::string data = read_file(path);
    ::unlink(path);
    size_t lines = 0;
    for (size_t pos = 0; pos < data.size();) {
        const size_t nl = data.find('\n', pos);
        CHECK(nl != std::string::npos);
        const std::string line = data.substr(pos, nl - pos);
        CHECK(line.size() >= 201 && line.size() <= 204);  // 200 filler + counter
        CHECK(line[0] == 'a' || line[0] == 'b');
        CHECK(line.find_first_not_of(line[0]) == 200);  // one uniform record
        ++lines;
        pos = nl + 1;
    }
    CHECK(lines == 4000);
    CHECK(data.find("\n\n") == std::string::npos);  // never an empty line
}

// ---- §6.10.3 CONNECT: the flags must describe the payload ----

struct ConnectInfo {
    uint8_t flags = 0;
    int keepalive = 0;
    std::string client_id, will_topic, will_payload, username, password;
    size_t consumed = 0;  // payload bytes the flags account for
    size_t payload_size = 0;
};

// Walks the variable header and payload strictly from the flag bits, exactly
// as a broker does. If the encoder sets different bits than the strings it
// appended, `consumed` stops short of `payload_size`.
ConnectInfo parse_connect(const std::string& body) {
    ConnectInfo ci;
    size_t pos = 0;
    auto rstr = [&](std::string& out) {
        CHECK(pos + 2 <= body.size());
        const size_t n =
            static_cast<size_t>((uint8_t(body[pos]) << 8) | uint8_t(body[pos + 1]));
        pos += 2;
        CHECK(pos + n <= body.size());
        out.assign(body, pos, n);
        pos += n;
    };
    std::string proto;
    rstr(proto);
    CHECK_STREQ(proto, "MQTT");
    CHECK(pos + 4 <= body.size());
    CHECK(uint8_t(body[pos]) == 4);  // protocol level 3.1.1
    ci.flags = uint8_t(body[pos + 1]);
    ci.keepalive = (uint8_t(body[pos + 2]) << 8) | uint8_t(body[pos + 3]);
    pos += 4;
    rstr(ci.client_id);
    if (ci.flags & 0x04) {
        rstr(ci.will_topic);
        rstr(ci.will_payload);
    }
    if (ci.flags & 0x80) rstr(ci.username);
    if (ci.flags & 0x40) rstr(ci.password);
    ci.consumed = pos;
    ci.payload_size = body.size();
    return ci;
}

// Minimal loopback listener: accepts one client, reads one packet, replies.
struct Loopback {
    int lfd = -1;
    int conn = -1;
    uint16_t port = 0;
    std::string rx;

    bool start() {
        lfd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (lfd < 0) return false;
        int one = 1;
        ::setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        if (::bind(lfd, reinterpret_cast<sockaddr*>(&a), sizeof a) != 0) return false;
        if (::listen(lfd, 4) != 0) return false;
        socklen_t sl = sizeof a;
        if (::getsockname(lfd, reinterpret_cast<sockaddr*>(&a), &sl) != 0) return false;
        port = ntohs(a.sin_port);
        return true;
    }

    bool accept_one(double deadline) {
        while (now_s() < deadline) {
            pollfd p{lfd, POLLIN, 0};
            if (::poll(&p, 1, 100) > 0) {
                conn = ::accept(lfd, nullptr, nullptr);
                return conn >= 0;
            }
        }
        return false;
    }

    bool read_packet(double deadline, uint8_t& type, std::string& body) {
        for (;;) {
            const size_t avail = rx.size();
            if (avail >= 2) {
                size_t len = 0, mult = 1, i = 1;
                bool header_done = false;
                while (true) {
                    if (i >= avail) break;
                    const uint8_t d = uint8_t(rx[i]);
                    len += size_t(d & 0x7f) * mult;
                    mult *= 128;
                    ++i;
                    if (!(d & 0x80)) {
                        header_done = true;
                        break;
                    }
                }
                if (header_done && avail >= i + len) {
                    type = uint8_t(uint8_t(rx[0]) >> 4);
                    body = rx.substr(i, len);
                    rx.erase(0, i + len);
                    return true;
                }
            }
            if (now_s() >= deadline) return false;
            pollfd p{conn, POLLIN, 0};
            if (::poll(&p, 1, 100) > 0) {
                char buf[1024];
                ssize_t n = ::recv(conn, buf, sizeof buf, 0);
                if (n <= 0) return false;
                rx.append(buf, size_t(n));
            }
        }
    }

    void send(const std::string& s) {
        if (conn >= 0) ::send(conn, s.data(), s.size(), 0);
    }
    void stop() {
        if (conn >= 0) ::close(conn);
        if (lfd >= 0) ::close(lfd);
        conn = -1;
        lfd = -1;
    }
};

uint16_t publish_pid(const std::string& body) {
    CHECK(body.size() >= 2);
    const size_t n = (size_t(uint8_t(body[0])) << 8) | uint8_t(body[1]);
    CHECK(body.size() >= 2 + n + 2);
    return static_cast<uint16_t>((uint8_t(body[2 + n]) << 8) |
                                 uint8_t(body[3 + n]));
}

void ack_pid(Loopback& lb, uint16_t pid, bool split = false) {
    std::string ack{"\x40\x02", 2};
    ack.push_back(static_cast<char>(pid >> 8));
    ack.push_back(static_cast<char>(pid & 0xff));
    if (!split) { lb.send(ack); return; }
    lb.send(ack.substr(0, 1));
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    lb.send(ack.substr(1));
}

void broker_handshake(Loopback& lb, vb::MqttLite& mqtt) {
    CHECK(lb.accept_one(now_s() + 5.0));
    uint8_t type = 0;
    std::string body;
    CHECK(lb.read_packet(now_s() + 5.0, type, body));
    CHECK(type == 1);
    lb.send(std::string("\x20\x02\x00\x00", 4));
}

void ack_online(Loopback& lb) {
    uint8_t type = 0;
    std::string body;
    CHECK(lb.read_packet(now_s() + 5.0, type, body));
    CHECK(type == 3);
    ack_pid(lb, publish_pid(body));
}

void test_confirmed_puback_and_drop_notifications() {
    Loopback lb;
    CHECK(lb.start());
    vb::MqttConfig cfg;
    cfg.host = "127.0.0.1"; cfg.port = lb.port; cfg.client_id = "vb-confirmed";
    cfg.topic_root = "R"; cfg.keepalive_s = 30.0;
    vb::MqttLite mqtt(cfg);
    std::string err;
    CHECK(mqtt.start(err));
    broker_handshake(lb, mqtt);
    ack_online(lb);

    std::atomic<int> ok{0}, failed{0};
    mqtt.publish_confirmed("events/one", "payload", false,
        [&](bool confirmed) { if (confirmed) ++ok; else ++failed; throw 1; });
    uint8_t type = 0; std::string body;
    CHECK(lb.read_packet(now_s() + 5.0, type, body));
    CHECK(type == 3);
    const uint16_t pid = publish_pid(body);
    ack_pid(lb, pid, true);
    const double deadline = now_s() + 2.0;
    while (ok.load() != 1 && now_s() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    CHECK(ok.load() == 1 && failed.load() == 0);
    // Unknown and duplicate PUBACKs do not produce another notification.
    ack_pid(lb, static_cast<uint16_t>(pid + 1));
    ack_pid(lb, pid);
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    CHECK(ok.load() == 1 && failed.load() == 0);
    mqtt.stop();
    lb.stop();

    vb::MqttLite offline(cfg);
    int evicted = 0, stopped = 0;
    for (size_t i = 0; i < vb::MqttLite::kQueueLimit + 1; ++i) {
        offline.publish_confirmed("events/drop", std::to_string(i), false,
            [&](bool confirmed) { if (confirmed) ++evicted; else ++stopped; });
    }
    CHECK(evicted == 0 && stopped == 1);
    offline.stop();
    CHECK(stopped == static_cast<int>(vb::MqttLite::kQueueLimit + 1));
}

void test_confirmed_reconnect_ack() {
    Loopback lb;
    CHECK(lb.start());
    vb::MqttConfig cfg;
    cfg.host = "127.0.0.1"; cfg.port = lb.port; cfg.client_id = "vb-reconnect";
    cfg.topic_root = "R"; cfg.keepalive_s = 30.0;
    cfg.reconnect_min_s = 0.05; cfg.reconnect_max_s = 0.05;
    vb::MqttLite mqtt(cfg);
    std::string err;
    CHECK(mqtt.start(err));
    broker_handshake(lb, mqtt);
    ack_online(lb);
    std::atomic<int> ok{0}, failed{0};
    mqtt.publish_confirmed("events/reconnect", "payload", false,
        [&](bool confirmed) { if (confirmed) ++ok; else ++failed; });
    uint8_t type = 0; std::string body;
    CHECK(lb.read_packet(now_s() + 5.0, type, body));
    CHECK(type == 3);
    ::close(lb.conn); lb.conn = -1;
    CHECK(lb.accept_one(now_s() + 8.0));
    CHECK(lb.read_packet(now_s() + 5.0, type, body));
    CHECK(type == 1);
    lb.send(std::string("\x20\x02\x00\x00", 4));
    ack_online(lb);
    CHECK(lb.read_packet(now_s() + 8.0, type, body));
    CHECK(type == 3);
    ack_pid(lb, publish_pid(body));
    const double deadline = now_s() + 2.0;
    while (ok.load() != 1 && now_s() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    CHECK(ok.load() == 1 && failed.load() == 0);

    // A second disconnect after the one permitted resend reports failure once.
    mqtt.publish_confirmed("events/exhaust", "payload", false,
        [&](bool confirmed) { if (confirmed) ++ok; else ++failed; });
    CHECK(lb.read_packet(now_s() + 5.0, type, body));
    CHECK(type == 3);
    ::close(lb.conn); lb.conn = -1;
    CHECK(lb.accept_one(now_s() + 8.0));
    CHECK(lb.read_packet(now_s() + 5.0, type, body));
    CHECK(type == 1);
    lb.send(std::string("\x20\x02\x00\x00", 4));
    ack_online(lb);
    CHECK(lb.read_packet(now_s() + 8.0, type, body));
    CHECK(type == 3);
    ::close(lb.conn); lb.conn = -1;
    const double failed_deadline = now_s() + 3.0;
    while (failed.load() != 1 && now_s() < failed_deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    CHECK(failed.load() == 1 && ok.load() == 1);
    mqtt.stop();
    lb.stop();
}

// Runs one CONNECT exchange against a loopback listener.
ConnectInfo connect_case(const char* username, const char* password,
                         double keepalive_s = 30.0) {
    Loopback lb;
    CHECK(lb.start());
    vb::MqttConfig cfg;
    cfg.host = "127.0.0.1";
    cfg.port = lb.port;
    cfg.client_id = "vb-test";
    cfg.topic_root = "R";
    cfg.keepalive_s = keepalive_s;
    if (username) cfg.username = username;
    if (password) cfg.password = password;
    vb::MqttLite mqtt(cfg);
    std::string err;
    CHECK(mqtt.start(err));
    CHECK(lb.accept_one(now_s() + 5.0));
    uint8_t type = 0;
    std::string body;
    CHECK(lb.read_packet(now_s() + 5.0, type, body));
    CHECK(type == 1);  // CONNECT
    const ConnectInfo ci = parse_connect(body);
    lb.send(std::string("\x20\x02\x00\x00", 4));  // CONNACK
    mqtt.stop();
    lb.stop();
    // keepalive reaches the wire as requested; the callers above rely on the
    // MqttConfig default (30), the boundary case asks for 1.
    CHECK(ci.keepalive == static_cast<int>(keepalive_s));
    CHECK_STREQ(ci.client_id, "vb-test");
    CHECK_STREQ(ci.will_topic, "R/status");
    // The flags must account for every payload byte. The old encoder kept the
    // flag byte at 0x2E while appending user name and password, so a broker
    // (or this parser) stopped after the will payload and left both strings
    // unread: a connection that "succeeded" with the credentials ignored.
    CHECK(ci.consumed == ci.payload_size);
    return ci;
}

void test_connect_flags_follow_credentials() {
    // LWT (QoS1, retain) and clean session are always on; credentials are not.
    const ConnectInfo anon = connect_case(nullptr, nullptr);
    CHECK(anon.flags == (0x02 | 0x04 | 0x08 | 0x20));
    CHECK((anon.flags & 0x80) == 0);
    CHECK((anon.flags & 0x40) == 0);
    CHECK(((anon.flags >> 3) & 3) == 1);  // will QoS1
    CHECK((anon.flags & 0x20) != 0);      // will retain
    CHECK((anon.flags & 0x02) != 0);      // clean session
    CHECK(anon.username.empty() && anon.password.empty());

    // User name only: bit 7 set, bit 6 clear.
    const ConnectInfo user = connect_case("alice", nullptr);
    CHECK(user.flags == (0x02 | 0x04 | 0x08 | 0x20 | 0x80));
    CHECK_STREQ(user.username, "alice");
    CHECK(user.password.empty());

    // User name + password: both bits set.
    const ConnectInfo both = connect_case("alice", "s3cret");
    CHECK(both.flags == (0x02 | 0x04 | 0x08 | 0x20 | 0x80 | 0x40));
    CHECK_STREQ(both.username, "alice");
    CHECK_STREQ(both.password, "s3cret");

    // Empty credentials mean no flag bit and no extra payload: a zero-length
    // user name would otherwise be announced and read as one string.
    const ConnectInfo empty = connect_case("", "");
    CHECK((empty.flags & 0x80) == 0);
    CHECK((empty.flags & 0x40) == 0);
}

// A password without a user name is invalid in MQTT 3.1.1 §3.1.2.9 and is
// refused at startup instead of being sent as an unparseable CONNECT.
void test_password_without_username_rejected() {
    vb::MqttConfig cfg;
    cfg.host = "127.0.0.1";
    cfg.port = 1;
    cfg.topic_root = "R";
    cfg.password = "s3cret";
    vb::MqttLite mqtt(cfg);
    std::string err;
    CHECK(!mqtt.start(err));
    CHECK(err.find("password requires username") != std::string::npos);
    mqtt.stop();
}

// keepalive_s is a config-schema key with minimum 1 (§6.6) and the schema is
// the contract, so a value that would switch the keepalive timer off is
// refused at startup instead of being sent as CONNECT keepalive=0. Disabling
// it would also drop the 1.5x-no-inbound dead-link detection (§6.7.1).
void test_keepalive_below_one_rejected() {
    for (double bad : {0.0, 0.5, -5.0}) {
        vb::MqttConfig cfg;
        cfg.host = "127.0.0.1";
        cfg.port = 1;
        cfg.topic_root = "R";
        cfg.keepalive_s = bad;
        vb::MqttLite mqtt(cfg);
        std::string err;
        CHECK(!mqtt.start(err));
        CHECK(err.find("keepalive_s must be >= 1") != std::string::npos);
        mqtt.stop();
    }
    // The boundary itself is valid and reaches the wire unchanged.
    const ConnectInfo one = connect_case(nullptr, nullptr, 1.0);
    CHECK(one.keepalive == 1);
}

}  // namespace

int main() {
    test_event_fields_recursive_rounding();
    test_ts_ms_half_to_even();
    test_box_inverse_uses_double();
    test_coordinates_clipped();
    test_mqtt_framing();
    test_jsonl_records_not_interleaved();
    test_connect_flags_follow_credentials();
    test_password_without_username_rejected();
    test_keepalive_below_one_rejected();
    test_confirmed_puback_and_drop_notifications();
    test_confirmed_reconnect_ack();
    std::fprintf(stderr, "test_out: all checks passed\n");
    return 0;
}
