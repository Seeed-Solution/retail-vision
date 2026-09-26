// MqttLite implementation (spec BASE-1 §6.10.3, M1.19). See mqtt_lite.h.
#include "out/mqtt_lite.h"

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <random>

#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#ifdef VB_WITH_TLS
#include <openssl/err.h>
#include <openssl/x509v3.h>
#include <openssl/ssl.h>
#endif

#include "vb/json.h"

namespace vb {

namespace {

double now_s() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}

void put_u16(std::string& s, uint16_t v) {
    s.push_back(static_cast<char>((v >> 8) & 0xff));
    s.push_back(static_cast<char>(v & 0xff));
}

void put_str(std::string& s, const std::string& v) {  // MQTT UTF-8 string
    put_u16(s, static_cast<uint16_t>(v.size()));
    s += v;
}

std::string remaining_length(size_t n) {
    std::string out;
    do {
        uint8_t d = n % 128;
        n /= 128;
        if (n > 0) d |= 0x80;
        out.push_back(static_cast<char>(d));
    } while (n > 0);
    return out;
}

}  // namespace

MqttLite::MqttLite(MqttConfig cfg) : cfg_(std::move(cfg)) {
    if (cfg_.client_id.empty()) cfg_.client_id = "vb-runtime";
    Json on, off;
    on["schema"] = "vb.status/1";
    on["online"] = true;
    off["schema"] = "vb.status/1";
    off["online"] = false;
    online_payload_ = json_dump(on);
    offline_payload_ = json_dump(off);
}

MqttLite::~MqttLite() { stop(); }

std::string MqttLite::status_topic() const { return cfg_.topic_root + "/status"; }

std::string MqttLite::offline_status_json() const {
    Json off = json_parse(offline_payload_);
    uint64_t d = dropped_.load(std::memory_order_relaxed);
    if (d > 0) off["mqtt_dropped"] = d;
    return json_dump(off);
}

bool MqttLite::start(std::string& err) {
    if (cfg_.host.empty() || cfg_.topic_root.empty()) {
        err = "mqtt: host and topic_root are required";
        return false;
    }
    if (cfg_.tls) {
#ifndef VB_WITH_TLS
        err = "tls not compiled in";
        return false;
#else
        if (!cfg_.cert_file.empty() != !cfg_.key_file.empty()) {
            err = "mqtt: cert_file and key_file must be given together";
            return false;
        }
#endif
    }
    running_ = true;
    thread_ = std::thread([this] { run(); });
    return true;
}

void MqttLite::stop() {
    if (!running_.exchange(false)) {
        if (thread_.joinable()) thread_.join();
        return;
    }
    if (thread_.joinable()) thread_.join();
    close_socket();
}

void MqttLite::publish(const std::string& topic_suffix, const std::string& payload,
                       int qos, bool retain) {
    std::lock_guard<std::mutex> lk(mu_);
    if (q_.size() >= 1024) {
        q_.pop_front();
        dropped_.fetch_add(1, std::memory_order_relaxed);
    }
    q_.push_back(Msg{cfg_.topic_root + "/" + topic_suffix, payload, qos, retain, false});
}

// ---- socket / TLS ----

bool MqttLite::connect_socket(std::string& err) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    std::string port = std::to_string(cfg_.port);
    if (::getaddrinfo(cfg_.host.c_str(), port.c_str(), &hints, &res) != 0 || !res) {
        err = "mqtt: cannot resolve " + cfg_.host;
        return false;
    }
    int fd = -1;
    for (addrinfo* ai = res; ai; ai = ai->ai_next) {
        fd = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        if (::connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) break;
        ::close(fd);
        fd = -1;
    }
    ::freeaddrinfo(res);
    if (fd < 0) {
        err = "mqtt: cannot connect to " + cfg_.host + ":" + port;
        return false;
    }
    int one = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    fd_ = fd;
#ifdef VB_WITH_TLS
    if (cfg_.tls) {
        SSL_CTX* ctx = ::SSL_CTX_new(::TLS_client_method());
        if (!ctx) {
            err = "mqtt: SSL_CTX_new failed";
            close_socket();
            return false;
        }
        ::SSL_CTX_set_min_proto_version(ctx, TLS1_2_VERSION);
        ::SSL_CTX_set_verify(ctx, SSL_VERIFY_PEER, nullptr);
        if (!cfg_.ca_file.empty() &&
            ::SSL_CTX_load_verify_locations(ctx, cfg_.ca_file.c_str(), nullptr) != 1) {
            err = "mqtt: cannot load ca_file: " + cfg_.ca_file;
            ::SSL_CTX_free(ctx);
            close_socket();
            return false;
        }
        if (!cfg_.cert_file.empty() &&
            ::SSL_CTX_use_certificate_chain_file(ctx, cfg_.cert_file.c_str()) != 1) {
            err = "mqtt: cannot load cert_file: " + cfg_.cert_file;
            ::SSL_CTX_free(ctx);
            close_socket();
            return false;
        }
        if (!cfg_.key_file.empty() &&
            ::SSL_CTX_use_PrivateKey_file(ctx, cfg_.key_file.c_str(), SSL_FILETYPE_PEM) != 1) {
            err = "mqtt: cannot load key_file: " + cfg_.key_file;
            ::SSL_CTX_free(ctx);
            close_socket();
            return false;
        }
        SSL* ssl = ::SSL_new(ctx);
        ::SSL_set_fd(ssl, fd_);
        // Hostname check: the chain alone only proves a trusted CA issued the
        // certificate, not that it was issued for cfg_.host.
        X509_VERIFY_PARAM* vp = ::SSL_get0_param(ssl);
        ::X509_VERIFY_PARAM_set_hostflags(vp, X509_CHECK_FLAG_NO_PARTIAL_WILDCARDS);
        unsigned char ipbuf[16];
        bool is_ip = ::inet_pton(AF_INET, cfg_.host.c_str(), ipbuf) == 1 ||
                     ::inet_pton(AF_INET6, cfg_.host.c_str(), ipbuf) == 1;
        int hok = is_ip ? ::X509_VERIFY_PARAM_set1_ip_asc(vp, cfg_.host.c_str())
                        : ::X509_VERIFY_PARAM_set1_host(vp, cfg_.host.c_str(), 0);
        if (!is_ip) ::SSL_set_tlsext_host_name(ssl, cfg_.host.c_str());
        if (hok != 1) {
            err = "mqtt: cannot set TLS hostname check for " + cfg_.host;
            ::SSL_free(ssl);
            ::SSL_CTX_free(ctx);
            close_socket();
            return false;
        }
        if (::SSL_connect(ssl) != 1) {
            unsigned long e = ::ERR_get_error();
            char buf[256];
            ::ERR_error_string_n(e, buf, sizeof buf);
            err = "mqtt: TLS handshake failed: " + std::string(buf);
            ::SSL_free(ssl);
            ::SSL_CTX_free(ctx);
            close_socket();
            return false;
        }
        long vr = ::SSL_get_verify_result(ssl);
        if (vr != X509_V_OK) {
            err = std::string("mqtt: server certificate verify failed: ") +
                  ::X509_verify_cert_error_string(vr);
            ::SSL_free(ssl);
            ::SSL_CTX_free(ctx);
            close_socket();
            return false;
        }
        ssl_ = ssl;
        ssl_ctx_ = ctx;
    }
#else
    (void)err;
    if (cfg_.tls) {  // unreachable without TLS; keeps the fields referenced
        (void)ssl_;
        (void)ssl_ctx_;
    }
#endif
    return true;
}

void MqttLite::close_socket() {
#ifdef VB_WITH_TLS
    if (ssl_) {
        ::SSL_shutdown(static_cast<SSL*>(ssl_));
        ::SSL_free(static_cast<SSL*>(ssl_));
        ssl_ = nullptr;
    }
    if (ssl_ctx_) {
        ::SSL_CTX_free(static_cast<SSL_CTX*>(ssl_ctx_));
        ssl_ctx_ = nullptr;
    }
#endif
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    connected_ = false;
}

void MqttLite::send_packet(const std::string& pkt) {
#ifdef VB_WITH_TLS
    if (ssl_) {
        ::SSL_write(static_cast<SSL*>(ssl_), pkt.data(), static_cast<int>(pkt.size()));
    } else
#endif
        ::send(fd_, pkt.data(), pkt.size(), 0);
    last_tx_s_ = now_s();
}

bool MqttLite::mqtt_connect(std::string& err) {
    if (!connect_socket(err)) return false;
    std::string vh, payload;
    put_str(vh, "MQTT");
    vh.push_back(static_cast<char>(4));  // protocol level 3.1.1
    // flags: clean session | will | will QoS1 | will retain
    vh.push_back(static_cast<char>(0x02 | 0x04 | 0x08 | 0x20));
    put_u16(vh, static_cast<uint16_t>(cfg_.keepalive_s));
    put_str(payload, cfg_.client_id);
    put_str(payload, status_topic());
    put_str(payload, offline_payload_);
    if (!cfg_.username.empty()) {
        put_str(payload, cfg_.username);
        put_str(payload, cfg_.password);
    }
    std::string pkt(1, static_cast<char>(0x10));
    pkt += remaining_length(vh.size() + payload.size());
    pkt += vh;
    pkt += payload;
    send_packet(pkt);
    // CONNACK (or any error/close within 10 s)
    double deadline = now_s() + 10.0;
    std::string buf;
    while (now_s() < deadline) {
        pollfd p{fd_, POLLIN, 0};
        int pr = ::poll(&p, 1, 200);
        if (pr <= 0) continue;
        char tmp[512];
        ssize_t n;
#ifdef VB_WITH_TLS
        if (ssl_)
            n = ::SSL_read(static_cast<SSL*>(ssl_), tmp, sizeof tmp);
        else
#endif
            n = ::recv(fd_, tmp, sizeof tmp, 0);
        if (n <= 0) {
            err = "mqtt: CONNACK not received (connection closed)";
            close_socket();
            return false;
        }
        buf.append(tmp, tmp + n);
        last_rx_s_ = now_s();
        if (buf.size() >= 4 && (buf[0] >> 4) == 2) {
            if (buf[3] == 0) {
                connected_ = true;
                return true;
            }
            err = "mqtt: connection refused, code " + std::to_string(int(buf[3]));
            close_socket();
            return false;
        }
    }
    err = "mqtt: CONNACK timeout";
    close_socket();
    return false;
}

std::string MqttLite::encode_publish(const Msg& m, uint16_t pid) {
    std::string vh;
    put_str(vh, m.topic);
    if (m.qos > 0) put_u16(vh, pid);
    uint8_t flags = 0x30 | (m.qos == 1 ? 0x02 : 0) | (m.retain ? 0x01 : 0);
    std::string pkt(1, static_cast<char>(flags));
    pkt += remaining_length(vh.size() + m.payload.size());
    pkt += vh;
    pkt += m.payload;
    return pkt;
}

void MqttLite::publish_now(const Msg& m) {
    if (m.qos == 0) {
        send_packet(encode_publish(m, 0));
        return;
    }
    uint16_t pid = next_pid_;
    next_pid_ = static_cast<uint16_t>(next_pid_ % 65535 + 1);
    send_packet(encode_publish(m, pid));
    inflight_[pid] = {m, now_s()};
}

bool MqttLite::take_and_publish(double) {
    // QoS1 in-flight cap 16 (§6.10.3); PUBACK timeouts are handled in
    // drain_and_read (disconnect + single resend).
    if (inflight_.size() >= 16) return true;
    Msg m;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (q_.empty()) return true;
        m = std::move(q_.front());
        q_.pop_front();
    }
    publish_now(m);
    return true;
}

bool MqttLite::drain_and_read(double now) {
    // PUBACK timeout (5 s): requeue every unconfirmed message at the queue
    // head (each resent once), then disconnect + reconnect (§6.10.3).
    for (const auto& kv : inflight_) {
        if (now - kv.second.second >= 5.0) return false;  // PUBACK timeout
    }
    pollfd p{fd_, POLLIN, 0};
    int pr = ::poll(&p, 1, 20);
    if (pr > 0 && (p.revents & (POLLIN | POLLHUP | POLLERR))) {
        char tmp[1024];
        ssize_t n;
#ifdef VB_WITH_TLS
        if (ssl_)
            n = ::SSL_read(static_cast<SSL*>(ssl_), tmp, sizeof tmp);
        else
#endif
            n = ::recv(fd_, tmp, sizeof tmp, 0);
        if (n <= 0) return false;  // connection lost
        last_rx_s_ = now;
        size_t pos = 0;
        while (pos + 2 <= static_cast<size_t>(n)) {
            uint8_t type = tmp[pos] >> 4;
            size_t len = 0, mult = 1, i = pos + 1;
            while (i < static_cast<size_t>(n)) {
                len += (tmp[i] & 0x7f) * mult;
                mult *= 128;
                bool cont = tmp[i] & 0x80;
                ++i;
                if (!cont) break;
            }
            if (type == 4 && len == 2 && pos + 4 <= static_cast<size_t>(n)) {
                uint16_t pid = (uint8_t(tmp[pos + 2]) << 8) | uint8_t(tmp[pos + 3]);
                inflight_.erase(pid);
            }
            pos = i + len;
        }
    }
    // Keepalive: PINGREQ after keepalive/2 idle; 1.5x keepalive without
    // inbound packets -> dead.
    double idle_tx = now - last_tx_s_, idle_rx = now - last_rx_s_;
    if (idle_tx > cfg_.keepalive_s / 2) send_packet(std::string("\xC0\x00", 2));
    if (idle_rx > cfg_.keepalive_s * 1.5) return false;
    return true;
}

void MqttLite::run() {
    std::mt19937 rng{std::random_device{}()};
    std::string err;
    while (running_) {
        if (!mqtt_connect(err)) {
            std::fprintf(stderr, "%s\n", err.c_str());
        } else {
            std::fprintf(stderr, "mqtt: connected to %s:%u\n", cfg_.host.c_str(),
                         static_cast<unsigned>(cfg_.port));
            backoff_s_ = 0;
            last_rx_s_ = last_tx_s_ = now_s();
            reconnects_.fetch_add(1, std::memory_order_relaxed);
            // Retained status online:true first after every (re)connect.
            publish_now(Msg{status_topic(), online_payload_, 1, true, false});
        }
        while (running_ && connected_) {
            double now = now_s();
            if (!take_and_publish(now)) break;
            if (!drain_and_read(now)) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        if (!running_) break;
        // Unconfirmed QoS1 messages go back to the queue head and are resent
        // once after the reconnect (§6.10.3).
        {
            std::lock_guard<std::mutex> lk(mu_);
            for (auto& kv : inflight_) {
                Msg m = kv.second.first;
                if (!m.retried) {
                    m.retried = true;
                    q_.push_front(m);
                }
            }
        }
        inflight_.clear();
        close_socket();
        // Backoff 1,2,4,...<=max, +-20% jitter.
        double base = backoff_s_ == 0 ? cfg_.reconnect_min_s
                                      : std::min(backoff_s_ * 2, cfg_.reconnect_max_s);
        double jitter = 1.0 + (std::uniform_real_distribution<double>(-0.2, 0.2)(rng));
        backoff_s_ = base;
        double wait = base * jitter;
        std::fprintf(stderr, "mqtt: reconnecting in %.2fs\n", wait);
        double deadline = now_s() + wait;
        while (running_ && now_s() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (running_) return;
    // Graceful shutdown: retained online:false, then DISCONNECT. If the
    // connection was down, make one final connect attempt for the offline
    // status (bounded by the CONNACK timeout).
    if (!connected_) {
        std::string e2;
        if (mqtt_connect(e2)) {
            reconnects_.fetch_add(1, std::memory_order_relaxed);
        }
    }
    if (connected_) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            q_.clear();
        }
        publish_now(Msg{status_topic(), offline_status_json(), 1, true, true});
        send_packet(std::string("\xE0\x00", 2));
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    close_socket();
}

}  // namespace vb
