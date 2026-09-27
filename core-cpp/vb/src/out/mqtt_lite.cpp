// MqttLite implementation (spec BASE-1 §6.10.3, M1.19). See mqtt_lite.h.
#include "out/mqtt_lite.h"

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

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

#include "out/mqtt_frame.h"
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

bool set_nonblocking(int fd) {
    int fl = ::fcntl(fd, F_GETFL, 0);
    if (fl < 0) return false;
    return ::fcntl(fd, F_SETFL, fl | O_NONBLOCK) == 0;
}

// Bounded wait for `events` on fd: never blocks longer than the deadline, and
// slices at 100 ms so a stop() request is picked up promptly.
bool poll_ready(int fd, short events, double deadline_s) {
    if (fd < 0) return false;
    while (now_s() < deadline_s) {
        pollfd p{fd, events, 0};
        int pr = ::poll(&p, 1, 100);
        if (pr > 0) return true;
        if (pr < 0 && errno != EINTR) return false;
    }
    return false;
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
    // MQTT 3.1.1 §3.1.2.9: the password flag (bit 6) may only be set when the
    // user name flag (bit 7) is. A password without a user name would make the
    // CONNECT payload unparseable for the broker, so it is refused here
    // instead of being sent as a malformed packet.
    if (cfg_.password.empty() == false && cfg_.username.empty()) {
        err = "mqtt: password requires username";
        return false;
    }
    // keepalive_s < 1 is refused instead of being read as MQTT 3.1.1's
    // "keepalive 0 = no keepalive": the schema says minimum 1 and the schema
    // is the contract (§6.6), and a disabled keepalive would also drop the
    // 1.5x-no-inbound rule that detects a silently dead link (§6.7.1).
    if (cfg_.keepalive_s < 1.0) {
        err = "mqtt: keepalive_s must be >= 1";
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
    // The worker drains its queue and publishes the final offline status first;
    // only if that overruns the grace budget is the socket woken with
    // shutdown(2) so a blocked connect/read/write cannot hold the join open.
    const double deadline = now_s() + kStopGraceS;
    while (!shutdown_done_.load(std::memory_order_acquire) && now_s() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    if (!shutdown_done_.load(std::memory_order_acquire)) wake_io();
    if (thread_.joinable()) thread_.join();
    close_socket();
}

void MqttLite::publish(const std::string& topic_suffix, const std::string& payload,
                       int qos, bool retain) {
    std::lock_guard<std::mutex> lk(mu_);
    evict_oldest_locked();  // bounded queue: full -> drop oldest, counted
    q_.push_back(Msg{cfg_.topic_root + "/" + topic_suffix, payload, qos, retain, false});
}

// Caller holds mu_.
void MqttLite::evict_oldest_locked() {
    if (q_.size() < kQueueLimit) return;
    q_.pop_front();
    dropped_.fetch_add(1, std::memory_order_relaxed);
}

void MqttLite::push_front_bounded(const Msg& m) {
    std::lock_guard<std::mutex> lk(mu_);
    // A QoS1 requeue goes to the head, but it must honour the same 1024-record
    // cap as publish(): pushing unconditionally let the queue grow past the
    // limit (measured 1040) and stay there.
    evict_oldest_locked();
    q_.push_front(m);
}

// ---- socket / TLS ----

bool MqttLite::wait_ready(short events, double deadline_s) {
    return poll_ready(fd_, events, deadline_s);
}

bool MqttLite::io_write(const char* p, size_t n, double deadline_s) {
    size_t off = 0;
    while (off < n) {
        ssize_t w;
#ifdef VB_WITH_TLS
        if (ssl_) {
            w = ::SSL_write(static_cast<SSL*>(ssl_), p + off,
                            static_cast<int>(n - off));
            if (w <= 0) {
                int e = ::SSL_get_error(static_cast<SSL*>(ssl_), static_cast<int>(w));
                if (e == SSL_ERROR_WANT_READ) {
                    if (!wait_ready(POLLIN, deadline_s)) return false;
                    continue;
                }
                if (e == SSL_ERROR_WANT_WRITE) {
                    if (!wait_ready(POLLOUT, deadline_s)) return false;
                    continue;
                }
                return false;
            }
            off += static_cast<size_t>(w);
            continue;
        }
#endif
        w = ::send(fd_, p + off, n - off, 0);
        if (w > 0) {
            off += static_cast<size_t>(w);
            continue;
        }
        if (w < 0 && errno == EINTR) continue;
        if (w < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            if (!wait_ready(POLLOUT, deadline_s)) return false;
            continue;
        }
        return false;  // EPIPE / ECONNRESET / EBADF / shutdown(2)
    }
    last_tx_s_ = now_s();
    return true;
}

int MqttLite::io_read() {
    char buf[4096];
    bool got = false;
    for (;;) {
        ssize_t n;
#ifdef VB_WITH_TLS
        if (ssl_) {
            n = ::SSL_read(static_cast<SSL*>(ssl_), buf, sizeof buf);
            if (n <= 0) {
                int e = ::SSL_get_error(static_cast<SSL*>(ssl_), static_cast<int>(n));
                if (e == SSL_ERROR_WANT_READ || e == SSL_ERROR_WANT_WRITE)
                    return got ? 1 : 0;
                return -1;
            }
            rxbuf_.append(buf, static_cast<size_t>(n));
            last_rx_s_ = now_s();
            got = true;
            continue;
        }
#endif
        n = ::recv(fd_, buf, sizeof buf, 0);
        if (n > 0) {
            rxbuf_.append(buf, static_cast<size_t>(n));
            last_rx_s_ = now_s();
            got = true;
            continue;
        }
        if (n == 0) return -1;  // peer closed
        if (errno == EINTR) continue;
        if (errno == EAGAIN || errno == EWOULDBLOCK) return got ? 1 : 0;
        return -1;
    }
}

int MqttLite::next_packet(uint8_t& type, std::string& body) {
    // MQTT framing lives in mqtt_frame.h (unit-tested in tests/test_out.cpp).
    return static_cast<int>(mqtt_take_packet(rxbuf_, type, body));
}

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
    const double deadline = now_s() + kConnectTimeoutS;
    int fd = -1;
    for (addrinfo* ai = res; ai; ai = ai->ai_next) {
        if (now_s() >= deadline) break;
        fd = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        if (!set_nonblocking(fd)) {
            ::close(fd);
            fd = -1;
            continue;
        }
        // Non-blocking connect with a deadline: a black-holed peer used to hold
        // the worker in a kernel-level connect() for minutes, which made
        // stop() hang as well.
        if (::connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) break;
        if (errno == EINPROGRESS && poll_ready(fd, POLLOUT, deadline)) {
            int so_err = 0;
            socklen_t slen = sizeof so_err;
            if (::getsockopt(fd, SOL_SOCKET, SO_ERROR, &so_err, &slen) == 0 &&
                so_err == 0)
                break;
        }
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
    {
        std::lock_guard<std::mutex> lk(fd_mu_);
        wake_fd_ = fd;  // wake_io() may now interrupt this connection's I/O
    }
    rxbuf_.clear();
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
        if (!cfg_.ca_file.empty()) {
            if (::SSL_CTX_load_verify_locations(ctx, cfg_.ca_file.c_str(), nullptr) != 1) {
                err = "mqtt: cannot load ca_file: " + cfg_.ca_file;
                ::SSL_CTX_free(ctx);
                close_socket();
                return false;
            }
        } else if (::SSL_CTX_set_default_verify_paths(ctx) != 1) {
            // §6.10.3: an empty ca_file means "use the system CA store". Without
            // this call SSL_VERIFY_PEER had no trust anchors at all, so every
            // broker using a system-trusted certificate failed the handshake.
            err = "mqtt: cannot load the system CA store";
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
        const double hs_deadline = now_s() + kConnectTimeoutS;
        for (;;) {
            int r = ::SSL_connect(ssl);
            if (r == 1) break;
            int e = ::SSL_get_error(ssl, r);
            if (e == SSL_ERROR_WANT_READ || e == SSL_ERROR_WANT_WRITE) {
                if (wait_ready(e == SSL_ERROR_WANT_READ ? POLLIN : POLLOUT, hs_deadline))
                    continue;
                err = "mqtt: TLS handshake timeout";
            } else {
                unsigned long ee = ::ERR_get_error();
                char buf[256];
                ::ERR_error_string_n(ee, buf, sizeof buf);
                err = "mqtt: TLS handshake failed: " + std::string(buf);
            }
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
    if (cfg_.tls) {  // start() refuses this; keeps the fields referenced
        (void)ssl_;
        (void)ssl_ctx_;
    }
#endif
    return true;
}

void MqttLite::close_socket() {
    // fd_mu_ also makes wake_io() safe: the block below either sees a live fd
    // or sees -1, never a descriptor that has been closed and recycled.
    std::lock_guard<std::mutex> lk(fd_mu_);
    wake_fd_ = -1;
#ifdef VB_WITH_TLS
    if (ssl_) {
        ::SSL_shutdown(static_cast<SSL*>(ssl_));  // non-blocking: never waits
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
    rxbuf_.clear();
    connected_ = false;
}

void MqttLite::wake_io() {
    std::lock_guard<std::mutex> lk(fd_mu_);
    if (wake_fd_ >= 0) ::shutdown(wake_fd_, SHUT_RDWR);
}

bool MqttLite::send_packet(const std::string& pkt) {
    if (fd_ < 0) return false;
    // Complete write loop: a short send() used to be ignored, so the remaining
    // bytes were never sent while the message was still recorded as in flight.
    return io_write(pkt.data(), pkt.size(), now_s() + kWriteTimeoutS);
}

bool MqttLite::mqtt_connect(std::string& err) {
    if (!connect_socket(err)) return false;
    std::string vh, payload;
    put_str(vh, "MQTT");
    vh.push_back(static_cast<char>(4));  // protocol level 3.1.1
    // flags: clean session | will | will QoS1 | will retain (+ user name /
    // password below). The credentials are in the payload only when their flag
    // bit is set, so the flags have to be derived from the same values: a
    // broker that needs authentication parsed the fixed 0x2E and never looked
    // for the user name that the payload carried.
    uint8_t flags = 0x02 | 0x04 | 0x08 | 0x20;
    if (!cfg_.username.empty()) flags |= 0x80;  // user name flag
    if (!cfg_.password.empty()) flags |= 0x40;  // password flag (needs bit 7)
    vh.push_back(static_cast<char>(flags));
    put_u16(vh, static_cast<uint16_t>(cfg_.keepalive_s));
    put_str(payload, cfg_.client_id);
    put_str(payload, status_topic());
    put_str(payload, offline_payload_);
    if (!cfg_.username.empty()) {
        put_str(payload, cfg_.username);
        if (!cfg_.password.empty()) put_str(payload, cfg_.password);
    }
    std::string pkt(1, static_cast<char>(0x10));
    pkt += remaining_length(vh.size() + payload.size());
    pkt += vh;
    pkt += payload;
    if (!send_packet(pkt)) {
        err = "mqtt: cannot send CONNECT";
        close_socket();
        return false;
    }
    // CONNACK (or any error/close within 10 s)
    double deadline = now_s() + 10.0;
    while (now_s() < deadline) {
        uint8_t type = 0;
        std::string body;
        int st = next_packet(type, body);
        if (st < 0) {
            err = "mqtt: malformed packet from broker";
            close_socket();
            return false;
        }
        if (st == 1) {
            if (type == 2 && body.size() >= 2) {
                uint8_t rc = static_cast<uint8_t>(body[1]);
                if (rc == 0) {
                    connected_ = true;
                    return true;
                }
                err = "mqtt: connection refused, code " + std::to_string(int(rc));
                close_socket();
                return false;
            }
            continue;  // not CONNACK: keep waiting for it
        }
        // Need more bytes: a CONNACK split across TCP segments is fine now, the
        // buffer is connection-level and only complete packets are consumed.
        if (!wait_ready(POLLIN, deadline)) continue;
        if (io_read() < 0) {
            err = "mqtt: CONNACK not received (connection closed)";
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

bool MqttLite::publish_now(const Msg& m) {
    if (m.qos == 0) return send_packet(encode_publish(m, 0));
    uint16_t pid = next_pid_;
    next_pid_ = static_cast<uint16_t>(next_pid_ % 65535 + 1);
    if (!send_packet(encode_publish(m, pid))) return false;
    inflight_[pid] = {m, now_s()};
    return true;
}

bool MqttLite::take_and_publish(double) {
    // QoS1 in-flight cap 16 (§6.10.3); PUBACK timeouts are handled in
    // drain_and_read (disconnect + single resend).
    if (inflight_.size() >= kInflightLimit) return true;
    Msg m;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (q_.empty()) return true;
        m = std::move(q_.front());
        q_.pop_front();
    }
    return publish_now(m);
}

bool MqttLite::drain_and_read(double now) {
    // PUBACK timeout (5 s): requeue every unconfirmed message at the queue
    // head (each resent once), then disconnect + reconnect (§6.10.3).
    for (const auto& kv : inflight_) {
        if (now - kv.second.second >= kPubackTimeoutS) return false;
    }
    // Read one bounded burst; the caller paces this loop, so the wait is short
    // enough to keep reconnect and shutdown latency in check.
    if (wait_ready(POLLIN, now + 0.02)) {
        if (io_read() < 0) return false;
    }
    // Parse every complete packet buffered so far. A PUBACK split across TCP
    // segments stays in rxbuf_ until its last byte arrives; the old code threw
    // the buffer away each round, so fragmented PUBACKs were never matched and
    // QoS1 messages were re-sent after every 5 s timeout.
    for (;;) {
        uint8_t type = 0;
        std::string body;
        const int st = next_packet(type, body);
        if (st < 0) return false;  // malformed or oversized -> drop the connection
        if (st == 0) break;
        if (type == 4 && body.size() >= 2) {  // PUBACK
            uint16_t pid =
                static_cast<uint16_t>((uint8_t(body[0]) << 8) | uint8_t(body[1]));
            inflight_.erase(pid);
        }
        // PINGRESP (13) and any other packet type need no action.
    }
    // Keepalive: PINGREQ after keepalive/2 idle; 1.5x keepalive without
    // inbound packets -> dead. keepalive_s is always >= 1 here (start()
    // refuses anything lower), so the timer and the liveness timeout are
    // unconditional; MQTT 3.1.1's "keepalive 0 = no keepalive" is not a
    // supported configuration (§6.6 schema minimum 1).
    {
        double idle_tx = now - last_tx_s_, idle_rx = now - last_rx_s_;
        if (idle_tx > cfg_.keepalive_s / 2) {
            if (!send_packet(std::string("\xC0\x00", 2))) return false;
        }
        if (idle_rx > cfg_.keepalive_s * 1.5) return false;
    }
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
            if (!publish_now(Msg{status_topic(), online_payload_, 1, true, false}))
                connected_ = false;
        }
        while (running_ && connected_) {
            double now = now_s();
            if (!take_and_publish(now)) break;
            if (!drain_and_read(now)) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        if (!running_) break;
        // Unconfirmed QoS1 messages go back to the queue head and are resent
        // once after the reconnect (§6.10.3). Reversed insertion keeps them in
        // pid order at the head, and each one passes the 1024-record cap.
        std::vector<Msg> requeue;
        for (auto& kv : inflight_) {
            if (kv.second.first.retried) continue;
            Msg m = kv.second.first;
            m.retried = true;
            requeue.push_back(m);
        }
        for (auto it = requeue.rbegin(); it != requeue.rend(); ++it)
            push_front_bounded(*it);
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
    // Graceful shutdown: drain what the caller queued (its own final
    // vb.status/1 among it), then DISCONNECT. If the connection was down, make
    // one final connect attempt for the offline status (bounded by the CONNACK
    // timeout).
    if (!connected_) {
        std::string e2;
        if (mqtt_connect(e2)) {
            reconnects_.fetch_add(1, std::memory_order_relaxed);
        }
    }
    if (connected_) {
        // §6.10.1: the queue is drained instead of cleared (business events
        // already handed to the publisher must not be thrown away), and the
        // last record is online:false. The caller's own final vb.status/1 — a
        // retained status message it queued before stopping — is held back and
        // published last: it carries the full stream/uptime snapshot, and a
        // bounded drain must not be able to push it out of the queue. The old
        // shutdown cleared the queue and then published a two-field stub.
        Msg final_status;
        bool have_final_status = false;
        {
            std::lock_guard<std::mutex> lk(mu_);
            for (auto it = q_.begin(); it != q_.end();) {
                if (it->retain && it->topic == status_topic()) {
                    final_status = *it;
                    have_final_status = true;
                    it = q_.erase(it);
                } else {
                    ++it;
                }
            }
        }
        const double drain_deadline = now_s() + kShutdownDrainS;
        for (;;) {
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (q_.empty()) break;
            }
            if (now_s() >= drain_deadline) break;
            if (!take_and_publish(now_s())) break;
            if (!drain_and_read(now_s())) break;
        }
        // Give the in-flight QoS1 messages a short window for their PUBACKs.
        const double ack_deadline = now_s() + 1.0;
        while (connected_ && !inflight_.empty() && now_s() < ack_deadline) {
            if (!drain_and_read(now_s())) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        if (have_final_status) {
            publish_now(final_status);
        } else {
            // Nothing was queued: publish the LWT-equivalent offline status so
            // subscribers still see online:false.
            publish_now(Msg{status_topic(), offline_status_json(), 1, true, true});
        }
        send_packet(std::string("\xE0\x00", 2));  // DISCONNECT
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    close_socket();
    shutdown_done_.store(true, std::memory_order_release);
}

}  // namespace vb
