// MQTT 3.1.1 fixed-header framing (spec BASE-1 §6.10.3, M1.19).
//
// Split out of mqtt_lite.cpp so the decoder is unit-testable without a socket:
// the transport (poll/SSL) needs a peer, the byte layout does not.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace vb {

enum class MqttFrame : int {
    Packet = 1,
    NeedMore = 0,
    Malformed = -1,
};

// Largest inbound packet this client accepts. It never subscribes, and the only
// packets it can legitimately receive are CONNACK/PUBACK/PINGRESP (4 bytes
// each). Remaining Length allows up to 268435455 bytes; announcing more than
// this is not the implemented protocol subset, so the caller drops the
// connection instead of buffering an attacker-sized packet.
inline constexpr size_t kMqttMaxInboundPacket = 65536;

// Takes one complete packet out of `buf`:
//   type = high nibble of the first byte, body = the Remaining Length bytes.
// On Packet exactly the packet's bytes are consumed, so a caller loop always
// makes progress. Malformed consumes nothing and means "drop the connection":
// it covers a Remaining Length longer than 4 bytes (§2.2.3), a fourth length
// byte that still has its continuation bit set, and an announced length above
// kMqttMaxInboundPacket.
//
// The 4-byte cap is what keeps the decoder honest: an unbounded decode of ten
// 0xFF bytes produced 2^64-11 and made `pos = i + len` wrap to 0, so the parse
// loop never advanced and pinned a core forever (and the stop() path with it).
inline MqttFrame mqtt_take_packet(std::string& buf, uint8_t& type,
                                  std::string& body) {
    const size_t avail = buf.size();
    if (avail < 2) return MqttFrame::NeedMore;
    uint32_t len = 0, mult = 1;
    size_t i = 1, digits = 0;
    for (;;) {
        if (i >= avail) return MqttFrame::NeedMore;  // header not complete yet
        const uint8_t d = static_cast<uint8_t>(buf[i]);
        len += uint32_t(d & 0x7f) * mult;  // mult <= 128^3, len <= 2^28-1: no overflow
        ++i;
        if (++digits == 4) {
            if (d & 0x80) return MqttFrame::Malformed;  // a 5th length byte
            break;
        }
        if (!(d & 0x80)) break;
        mult *= 128;
    }
    if (len > kMqttMaxInboundPacket) return MqttFrame::Malformed;
    const size_t total = i + static_cast<size_t>(len);  // bounded: no wraparound
    if (avail < total) return MqttFrame::NeedMore;      // body split across reads
    type = static_cast<uint8_t>(static_cast<uint8_t>(buf[0]) >> 4);
    body.assign(buf, i, len);
    buf.erase(0, total);
    return MqttFrame::Packet;
}

}  // namespace vb
