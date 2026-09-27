// MqttLite: hand-written MQTT 3.1.1 publisher subset, no third-party
// dependencies (spec BASE-1 §6.10.3, M1.19).
//
//   CONNECT (clean session, LWT = <status topic> online:false, QoS1 retain),
//   PUBLISH QoS0/qos1, PUBACK, PINGREQ/PINGRESP, DISCONNECT. No SUBSCRIBE.
//
//   - TCP_NODELAY on the socket.
//   - Bounded outbound queue (1024); full -> drop oldest, counted.
//   - QoS1 in-flight cap 16; PUBACK timeout 5 s -> reconnect and re-send
//     once (duplicates possible; receivers dedupe by ts_ms+seq+stream_id).
//   - Reconnect backoff 1,2,4,...<=60 s with +-20% jitter; after every
//     (re)connect the retained status online:true is published first.
//   - Keepalive PINGREQ; 1.5x keepalive without inbound traffic -> reconnect.
//     keepalive_s < 1 is refused at start(): the config schema minimum is 1,
//     and a disabled keepalive would also lose the dead-link detection that
//     §6.7.1 relies on. See drain_and_read().
//   - Optional TLS (VB_WITH_TLS, OpenSSL >= 1.1.1): TLS 1.2+, server cert
//     verified against ca_file (empty = system CA), optional client cert.
//     No skip-verify switch. Without TLS compiled in, tls=true fails
//     startup with "tls not compiled in".
//   - Every socket wait is deadline-bounded and the socket is non-blocking, so
//     stop() cannot hang on a dead peer: it first wakes any blocked I/O with
//     shutdown(2), then joins.
#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <thread>

namespace vb {

struct MqttConfig {
    std::string host;
    uint16_t port = 1883;
    std::string client_id;
    std::string username;
    std::string password;
    std::string topic_root;
    double keepalive_s = 30;
    bool tls = false;
    std::string ca_file, cert_file, key_file;
    // Test hook: shorten the reconnect schedule (defaults follow §6.10.3).
    double reconnect_min_s = 1.0;
    double reconnect_max_s = 60.0;
};

class MqttLite {
public:
    explicit MqttLite(MqttConfig cfg);
    ~MqttLite();

    // Validates config (incl. "tls not compiled in") and starts the worker
    // thread, which connects in the background.
    bool start(std::string& err);

    // Publishes retained online:false, sends DISCONNECT, joins the thread.
    void stop();

    // topic is the suffix after <topic_root>/ (e.g. "events/cam-a").
    void publish(const std::string& topic_suffix, const std::string& payload,
                 int qos, bool retain);

    uint64_t dropped() const { return dropped_.load(std::memory_order_relaxed); }
    bool connected() const { return connected_.load(std::memory_order_relaxed); }
    uint64_t reconnects() const { return reconnects_.load(std::memory_order_relaxed); }

    // Bounded outbound queue (§6.10.3): 1024 records, oldest dropped and
    // counted when full.
    static constexpr size_t kQueueLimit = 1024;
    // QoS1 in-flight cap (§6.10.3).
    static constexpr size_t kInflightLimit = 16;
    // PUBACK timeout (§6.10.3).
    static constexpr double kPubackTimeoutS = 5.0;
    // Connect/handshake deadline: bounds a black-holed TCP connect or TLS
    // handshake so stop() cannot hang on them.
    static constexpr double kConnectTimeoutS = 5.0;
    // Deadline for one write burst (all bytes of a packet).
    static constexpr double kWriteTimeoutS = 5.0;
    // Graceful-shutdown budget for draining the outbound queue.
    static constexpr double kShutdownDrainS = 1.5;
    // stop() waits this long for the graceful shutdown before force-waking it.
    static constexpr double kStopGraceS = 5.0;

private:
    struct Msg {
        std::string topic, payload;
        int qos = 0;
        bool retain = false;
        bool retried = false;  // resent once after a PUBACK timeout -> drop on next failure
    };

    void run();
    std::string offline_status_json() const;
    bool connect_socket(std::string& err);
    bool mqtt_connect(std::string& err);  // TCP (+TLS) + CONNECT/CONNACK
    void close_socket();
    bool send_packet(const std::string& pkt);  // full write loop, false on loss
    void wake_io();  // shutdown(2) on the live fd: unblocks any pending wait
    // Returns false when the connection was lost (and should be re-made).
    bool drain_and_read(double now_s);
    bool take_and_publish(double now_s);
    bool publish_now(const Msg& m);
    void push_front_bounded(const Msg& m);  // requeue with the 1024 cap applied
    void evict_oldest_locked();             // caller holds mu_: drop + count
    // Full write loop; false on error or deadline.
    bool io_write(const char* p, size_t n, double deadline_s);
    // Non-blocking read of everything available into rxbuf_.
    // 1 = bytes read, 0 = nothing available, -1 = connection lost.
    int io_read();
    // Waits until fd_ is ready for `events` or `deadline_s` passes.
    bool wait_ready(short events, double deadline_s);
    // One complete packet from rxbuf_ (consuming exactly its bytes).
    // 1 = packet, 0 = need more bytes, -1 = protocol violation.
    int next_packet(uint8_t& type, std::string& body);
    // True when a packet is only accepted after a full remaining length and
    // body are buffered (§6.10.3 PUBACK fragmentation).
    std::string encode_publish(const Msg& m, uint16_t pid);
    std::string status_topic() const;

    MqttConfig cfg_;
    std::string online_payload_;   // retained vb.status/1 online:true
    std::string offline_payload_;  // vb.status/1 online:false (LWT + stop)
    int fd_ = -1;
    void* ssl_ = nullptr;    // SSL* under VB_WITH_TLS
    void* ssl_ctx_ = nullptr;
    std::string rxbuf_;      // connection-level receive buffer
    std::thread thread_;
    std::mutex mu_;
    std::deque<Msg> q_;
    std::map<uint16_t, std::pair<Msg, double>> inflight_;  // pid -> (msg, sent_at)
    uint16_t next_pid_ = 1;
    double backoff_s_ = 0;      // 0 -> connect attempt hasn't failed yet
    double last_rx_s_ = 0, last_tx_s_ = 0;
    std::atomic<bool> running_{false};
    std::atomic<bool> shutdown_done_{false};
    std::atomic<uint64_t> dropped_{0};
    std::atomic<uint64_t> reconnects_{0};
    std::atomic<bool> connected_{false};
    // Guards fd_ lifetime against wake_io(): the close and the shutdown(2) run
    // under the same lock so stop() can never signal a recycled descriptor.
    std::mutex fd_mu_;
    int wake_fd_ = -1;
};

}  // namespace vb
