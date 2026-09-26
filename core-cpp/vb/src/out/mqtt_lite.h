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
//   - Optional TLS (VB_WITH_TLS, OpenSSL >= 1.1.1): TLS 1.2+, server cert
//     verified against ca_file (empty = system CA), optional client cert.
//     No skip-verify switch. Without TLS compiled in, tls=true fails
//     startup with "tls not compiled in".
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
    void send_packet(const std::string& pkt);
    // Returns false when the connection was lost (and should be re-made).
    bool drain_and_read(double now_s);
    bool take_and_publish(double now_s);
    void publish_now(const Msg& m);
    std::string encode_publish(const Msg& m, uint16_t pid);
    std::string status_topic() const;

    MqttConfig cfg_;
    std::string online_payload_;   // retained vb.status/1 online:true
    std::string offline_payload_;  // vb.status/1 online:false (LWT + stop)
    int fd_ = -1;
    void* ssl_ = nullptr;    // SSL* under VB_WITH_TLS
    void* ssl_ctx_ = nullptr;
    std::thread thread_;
    std::mutex mu_;
    std::deque<Msg> q_;
    std::map<uint16_t, std::pair<Msg, double>> inflight_;  // pid -> (msg, sent_at)
    uint16_t next_pid_ = 1;
    double backoff_s_ = 0;      // 0 -> connect attempt hasn't failed yet
    double last_rx_s_ = 0, last_tx_s_ = 0;
    std::atomic<bool> running_{false};
    std::atomic<uint64_t> dropped_{0};
    std::atomic<uint64_t> reconnects_{0};
    std::atomic<bool> connected_{false};
};

}  // namespace vb
