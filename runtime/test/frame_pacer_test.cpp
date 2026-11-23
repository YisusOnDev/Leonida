// FramePacer (src/frame_pacer.h) on a fake clock: the interval, the origin event with an offset, late waits, timeouts,
// the latency limit, tokens and their validation, statistics, and suspend/resume with re-registration. No GPU.
#include "../src/frame_pacer.h"

#include <chrono>
#include <cstdio>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <thread>

using namespace leonida;

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

constexpr std::uint32_t period = 16000;  // us; round numbers keep the arithmetic readable

// Every wait oversleeps: it wakes `oversleep_us` after the deadline it was given. `on_wake` (when set) runs at the
// wake with the pacer's mutex released, like another thread that gets the mutex before the waiter takes it back.
class DelayedWakeClock final : public FrameClock {
public:
    explicit DelayedWakeClock(std::uint64_t oversleep_us = period) : oversleep_us_(oversleep_us) {}
    std::uint64_t now_us() override { return now_; }
    void wait(std::unique_lock<std::mutex>& lock, std::condition_variable&, std::uint64_t deadline_us) override {
        now_ = deadline_us + oversleep_us_;
        if (!on_wake) return;
        lock.unlock();
        on_wake();
        lock.lock();
    }
    void set_now_us(std::uint64_t now) { now_ = now; }
    std::function<void()> on_wake;

private:
    std::uint64_t now_ = 0;
    std::uint64_t oversleep_us_;
};

struct Result {
    std::int32_t hr;
    std::uint64_t token;
};

// Runs a wait on its own thread.
std::future<Result> wait_async(FramePacer& pacer, std::uint32_t timeout_ms) {
    return std::async(std::launch::async, [&pacer, timeout_ms] {
        Result r{0, 0};
        r.hr = pacer.wait_frame_event(frame_event_origin, timeout_ms, 0, &r.token);
        return r;
    });
}

// True when the wait is still blocked after real time has had a chance to let it through.
bool blocked(std::future<Result>& f) {
    return f.wait_for(std::chrono::milliseconds(40)) == std::future_status::timeout;
}

// True when the wait has returned; otherwise the pacer is suspended, which lets the waiter go, so that a failure ends
// the test instead of hanging it on the fake clock.
bool finishes(std::future<Result>& f, FramePacer& pacer) {
    if (f.wait_for(std::chrono::seconds(10)) == std::future_status::ready) return true;
    pacer.suspend();
    return false;
}

// Display time of `token`'s frame, or no_deadline when the pacer has none (not shown, or replaced and never shown).
std::uint64_t display_time(FramePacer& pacer, std::uint64_t token) {
    std::uint32_t count = 1;
    FrameStatistic stat{};
    if (pacer.get_frame_statistics(token, stat_display, &count, &stat) != pacer_hr::ok || count != 1)
        return no_deadline;
    return stat.time_us;
}

bool setup(FramePacer& pacer, std::uint32_t latency, std::uint32_t offset = 0) {
    return pacer.set_frame_interval(period, latency, 0) == pacer_hr::ok &&
           pacer.schedule_frame_event(frame_event_origin, offset, 0) == pacer_hr::ok;
}

// Latency 1 with a 10 ms period: token 1 at origin 1 (10 ms), presented late at 21 ms so it shows at 30 ms. A wait
// from 22 ms with `timeout_ms` finds origin 2 (20 ms) due but the slot taken, and its clock wakes 35 ms in. It may
// only succeed when the slot was free by its deadline (22 ms + timeout); otherwise it times out with the token
// untouched, and a poll at 35 ms gets token 2. With `stats_on_wake`, a GetFrameStatisticsX call runs at the wake
// before the waiter has its mutex back, and retires frame 1 first: the wait must still judge the slot by the time
// the frame showed (30 ms), not by finding the slot free.
int late_slot_wait(std::uint32_t timeout_ms, std::uint64_t oversleep_us, std::int32_t expected,
                   bool stats_on_wake = false) {
    DelayedWakeClock clock(oversleep_us);
    FramePacer pacer(clock);
    FrameStatistic display;
    std::int32_t stats_hr = pacer_hr::false_;
    if (stats_on_wake) clock.on_wake = [&] {
        std::uint32_t count = 1;
        stats_hr = pacer.get_frame_statistics(1, stat_display, &count, &display);
    };
    EXPECT(pacer.set_frame_interval(10000, 1, 0) == pacer_hr::ok);
    EXPECT(pacer.schedule_frame_event(frame_event_origin, 0, 0) == pacer_hr::ok);
    clock.set_now_us(10000);
    std::uint64_t token = 0;
    EXPECT(pacer.wait_frame_event(frame_event_origin, 0, 0, &token) == pacer_hr::ok && token == 1);
    clock.set_now_us(21000);
    EXPECT(pacer.check_present_token(1) == pacer_hr::ok);
    pacer.commit_present_token(1, 1);
    EXPECT(pacer.frames_in_flight() == 1);
    clock.set_now_us(22000);
    token = 99;
    const std::int32_t hr = pacer.wait_frame_event(frame_event_origin, timeout_ms, 0, &token);
    EXPECT(clock.now_us() == 35000);  // the wait slept, and woke late
    if (stats_on_wake) EXPECT(stats_hr == pacer_hr::ok && display.time_us == 30000);
    EXPECT(hr == expected);
    if (expected == pacer_hr::timeout) {
        EXPECT(token == 99 && pacer.tokens_issued() == 1);
        EXPECT(pacer.wait_frame_event(frame_event_origin, 0, 0, &token) == pacer_hr::ok);
    }
    EXPECT(token == 2 && pacer.tokens_issued() == 2 && pacer.frames_in_flight() == 0);
    return 0;
}

}  // namespace

int main() {
    // Arguments and state errors.
    {
        ManualFrameClock clock;
        FramePacer pacer(clock);
        std::uint64_t token = 0;
        EXPECT(pacer.wait_frame_event(frame_event_origin, 0, 0, &token) == pacer_hr::invalid_state);  // nothing set
        EXPECT(pacer.schedule_frame_event(frame_event_origin, 0, 0) == pacer_hr::invalid_state);      // no interval
        EXPECT(pacer.set_frame_interval(0, 1, 0) == pacer_hr::invalid_arg);
        EXPECT(pacer.set_frame_interval(period, 1, 0) == pacer_hr::ok);
        EXPECT(pacer.wait_frame_event(frame_event_origin, 0, 0, &token) == pacer_hr::invalid_state);  // not scheduled
        EXPECT(pacer.schedule_frame_event(7, 0, 0) == pacer_hr::invalid_arg);                         // unknown event
        EXPECT(pacer.schedule_frame_event(frame_event_origin, period, 0) == pacer_hr::invalid_arg);   // offset >= period
        EXPECT(pacer.schedule_frame_event(frame_event_origin, 0, 0) == pacer_hr::ok);
        EXPECT(pacer.wait_frame_event(3, 0, 0, &token) == pacer_hr::invalid_arg);
        EXPECT(pacer.wait_frame_event(frame_event_origin, 0, 0, nullptr) == pacer_hr::invalid_arg);
        EXPECT(pacer.registered());
        // Setting the interval again keeps the scheduled event armed (it is re-based, not dropped).
        EXPECT(pacer.set_frame_interval(period, 1, 0) == pacer_hr::ok);
        EXPECT(pacer.registered());
    }

    // The origin event: each wait returns at the next vblank, with the next token.
    {
        ManualFrameClock clock;
        FramePacer pacer(clock);
        EXPECT(setup(pacer, 1));
        auto first = wait_async(pacer, infinite_timeout_ms);
        EXPECT(blocked(first));
        clock.advance_us(period - 1);
        EXPECT(blocked(first));
        clock.advance_us(1);
        Result r = first.get();
        EXPECT(r.hr == pacer_hr::ok && r.token == 1);
        auto second = wait_async(pacer, infinite_timeout_ms);
        EXPECT(blocked(second));
        clock.advance_us(period);
        r = second.get();
        EXPECT(r.hr == pacer_hr::ok && r.token == 2);
        EXPECT(pacer.tokens_issued() == 2);
    }

    {
        ManualFrameClock clock;
        auto tokens = std::make_shared<FrameTokenSource>();
        FramePacer pacer(clock, tokens);
        EXPECT(pacer.token_source() == tokens);
        const std::uint64_t outside = tokens->issue();
        EXPECT(outside == 1 && pacer.tokens_issued() == 1);
        EXPECT(pacer.check_present_token(outside) == pacer_hr::ok);
        EXPECT(pacer.check_present_token(outside + 1) == pacer_hr::invalid_arg);
        EXPECT(setup(pacer, 1));
        clock.advance_us(period);
        std::uint64_t token = 0;
        EXPECT(pacer.wait_frame_event(frame_event_origin, 0, 0, &token) == pacer_hr::ok && token == outside + 1);
        EXPECT(tokens->issue() == outside + 2);
        EXPECT(pacer.check_present_token(outside + 2) == pacer_hr::ok);
        EXPECT(pacer.check_present_token(outside + 3) == pacer_hr::invalid_arg);
    }

    // An offset delays the event inside the period: origin n comes at n * period + offset.
    {
        ManualFrameClock clock;
        FramePacer pacer(clock);
        EXPECT(setup(pacer, 1, 5000));
        auto wait = wait_async(pacer, infinite_timeout_ms);
        clock.advance_us(period);
        EXPECT(blocked(wait));
        clock.advance_us(4999);
        EXPECT(blocked(wait));
        clock.advance_us(1);
        EXPECT(wait.get().hr == pacer_hr::ok);
        std::uint32_t count = 1;
        FrameStatistic stat;
        EXPECT(pacer.get_frame_statistics(1, stat_frame_event, &count, &stat) == pacer_hr::ok);
        EXPECT(count == 1 && stat.type == stat_frame_event && stat.time_us == period + 5000);
    }

    // Scheduling later than the interval was set makes the next vblank the first origin, not the missed ones.
    {
        ManualFrameClock clock;
        FramePacer pacer(clock);
        EXPECT(pacer.set_frame_interval(period, 1, 0) == pacer_hr::ok);
        clock.advance_us(period * 5 + 100);
        EXPECT(pacer.schedule_frame_event(frame_event_origin, 0, 0) == pacer_hr::ok);
        auto wait = wait_async(pacer, infinite_timeout_ms);
        EXPECT(blocked(wait));
        clock.advance_us(period - 100);
        EXPECT(wait.get().token == 1);
    }

    // A late wait returns at once with the newest passed origin; the missed ones are skipped. Tokens still count
    // one per wait.
    {
        ManualFrameClock clock;
        FramePacer pacer(clock);
        EXPECT(setup(pacer, 1));
        clock.advance_us(period * 3 + period / 2);
        std::uint64_t token = 0;
        EXPECT(pacer.wait_frame_event(frame_event_origin, 0, 0, &token) == pacer_hr::ok && token == 1);
        std::uint32_t count = 1;
        FrameStatistic stat;
        EXPECT(pacer.get_frame_statistics(1, stat_frame_event, &count, &stat) == pacer_hr::ok);
        EXPECT(stat.time_us == period * 3);  // origin 3, not 1
        auto next = wait_async(pacer, infinite_timeout_ms);
        EXPECT(blocked(next));
        clock.advance_us(period / 2);
        EXPECT(next.get().token == 2);
    }

    // Timeouts: 0 polls, a finite one counts fake time, the token is left alone, and a later wait still works.
    {
        ManualFrameClock clock;
        FramePacer pacer(clock);
        EXPECT(setup(pacer, 1));
        std::uint64_t token = 99;
        EXPECT(pacer.wait_frame_event(frame_event_origin, 0, 0, &token) == pacer_hr::timeout && token == 99);
        auto wait = wait_async(pacer, 5);  // 5 ms = 5000 us of fake time
        EXPECT(blocked(wait));
        clock.advance_us(4999);
        EXPECT(blocked(wait));
        clock.advance_us(1);
        Result r = wait.get();
        EXPECT(r.hr == pacer_hr::timeout && r.token == 0);
        EXPECT(pacer.tokens_issued() == 0);
        auto again = wait_async(pacer, 100);
        EXPECT(blocked(again));
        clock.advance_us(period - 5000);  // the vblank, 11 ms after the first wait gave up
        r = again.get();
        EXPECT(r.hr == pacer_hr::ok && r.token == 1);
        // Poll with the origin due.
        clock.advance_us(period);
        EXPECT(pacer.wait_frame_event(frame_event_origin, 0, 0, &token) == pacer_hr::ok && token == 2);
    }

    // A delayed wake cannot let an origin scheduled after the deadline satisfy the wait.
    {
        DelayedWakeClock clock;
        FramePacer pacer(clock);
        EXPECT(setup(pacer, 1));
        std::uint64_t token = 99;
        EXPECT(pacer.wait_frame_event(frame_event_origin, 10, 0, &token) == pacer_hr::timeout);
        EXPECT(token == 99 && pacer.tokens_issued() == 0);
        EXPECT(pacer.wait_frame_event(frame_event_origin, 0, 0, &token) == pacer_hr::ok && token == 1);
    }

    // A delayed wake cannot let a slot that frees after the deadline satisfy the wait (late_slot_wait above).
    EXPECT(late_slot_wait(5, 8000, pacer_hr::timeout) == 0);  // deadline 27 ms, slot free at 30 ms, wake at 35 ms
    EXPECT(late_slot_wait(8, 5000, pacer_hr::ok) == 0);       // slot free exactly at the deadline (30 ms)
    EXPECT(late_slot_wait(10, 5000, pacer_hr::ok) == 0);      // free at 30 ms, deadline 32 ms; the late wake is fine
    // The same, with another call retiring frame 1 between the wake and the waiter taking its mutex back.
    EXPECT(late_slot_wait(5, 8000, pacer_hr::timeout, true) == 0);
    EXPECT(late_slot_wait(8, 5000, pacer_hr::ok, true) == 0);
    EXPECT(late_slot_wait(10, 5000, pacer_hr::ok, true) == 0);

    // Latency 1: a frame presented late is still in flight at the next origin, so the wait holds until it is shown.
    {
        ManualFrameClock clock;
        FramePacer pacer(clock);
        EXPECT(setup(pacer, 1));
        clock.advance_us(period);
        std::uint64_t token = 0;
        EXPECT(pacer.wait_frame_event(frame_event_origin, 0, 0, &token) == pacer_hr::ok && token == 1);
        clock.advance_us(period + period / 2);  // frame 1 is presented 1.5 vblanks late: now = 2.5 periods
        EXPECT(pacer.check_present_token(1) == pacer_hr::ok);
        pacer.commit_present_token(1, 1);
        EXPECT(pacer.frames_in_flight() == 1);
        auto wait = wait_async(pacer, infinite_timeout_ms);  // origin 2 has passed, but one frame is pending
        EXPECT(blocked(wait));
        clock.advance_us(period / 2 - 1);
        EXPECT(blocked(wait));
        clock.advance_us(1);  // vblank 3: frame 1 shows, the wait goes through with the origin of that vblank
        EXPECT(wait.get().token == 2);
        EXPECT(pacer.frames_in_flight() == 0);
        std::uint32_t count = 2;
        FrameStatistic stats[2];
        EXPECT(pacer.get_frame_statistics(1, stat_present | stat_display, &count, stats) == pacer_hr::ok);
        EXPECT(count == 2 && stats[0].type == stat_present && stats[0].time_us == period * 5 / 2 &&
               stats[1].type == stat_display && stats[1].time_us == period * 3);
    }

    // Latency 2 lets two frames queue; a third wait is held until the oldest has been shown.
    {
        ManualFrameClock clock;
        FramePacer pacer(clock);
        EXPECT(setup(pacer, 2));
        std::uint64_t tokens[3] = {};
        for (int i = 0; i < 2; ++i) {
            clock.advance_us(period);
            EXPECT(pacer.wait_frame_event(frame_event_origin, 0, 0, &tokens[i]) == pacer_hr::ok);
        }
        // Now = 2 periods; the app is slow and presents both frames at 3.5 periods: they show at vblanks 4 and 5.
        clock.advance_us(period * 3 / 2);
        EXPECT(pacer.check_present_token(tokens[0]) == pacer_hr::ok);
        pacer.commit_present_token(tokens[0], 1);
        EXPECT(pacer.check_present_token(tokens[1]) == pacer_hr::ok);
        pacer.commit_present_token(tokens[1], 1);
        EXPECT(pacer.frames_in_flight() == 2);
        auto wait = wait_async(pacer, infinite_timeout_ms);  // origin 3 has passed, but both slots are taken
        EXPECT(blocked(wait));
        clock.advance_us(period / 2);  // vblank 4 shows frame 1: one in flight, the wait goes through
        Result r = wait.get();
        EXPECT(r.hr == pacer_hr::ok && r.token == 3);
        EXPECT(pacer.frames_in_flight() == 1);
    }

    // Presents beyond the latency while the event is armed (several presents per wait): at most `latency` frames are
    // in flight. A present into a full queue replaces the newest queued frame (mailbox) and the older ones keep their
    // vblanks, so the next wait passes when the oldest is shown, within one period and not 999. Latency 1, 2 and 3.
    for (std::uint32_t latency = 1; latency <= 3; ++latency) {
        ManualFrameClock clock;
        FramePacer pacer(clock);
        EXPECT(setup(pacer, latency));
        clock.advance_us(period);  // origin 1
        std::uint64_t token = 0;
        EXPECT(pacer.wait_frame_event(frame_event_origin, 0, 0, &token) == pacer_hr::ok && token == 1);

        // 1000 blank presents with no wait between them: they queue for vblanks 2 .. latency + 1, and no further.
        for (int i = 0; i < 1000; ++i) pacer.note_untokened_present();
        EXPECT(pacer.frames_in_flight() == latency);
        auto wait = wait_async(pacer, infinite_timeout_ms);
        EXPECT(blocked(wait));
        clock.advance_us(period);  // vblank 2 shows the oldest: the wait passes with origin 2, not 999 periods later
        EXPECT(finishes(wait, pacer));
        Result r = wait.get();
        EXPECT(r.hr == pacer_hr::ok && r.token == 2);
        EXPECT(pacer.frames_in_flight() == latency - 1);
        clock.advance_us(std::uint64_t(period) * (latency - 1));  // vblank latency + 1 shows the newest
        EXPECT(pacer.frames_in_flight() == 0);

        // Tokened frames: latency + 1 tokens handed out with nothing in flight, then all presented at once.
        std::uint64_t tokens[4] = {};
        for (std::uint32_t i = 0; i <= latency; ++i) {
            clock.advance_us(period);
            EXPECT(pacer.wait_frame_event(frame_event_origin, 0, 0, &tokens[i]) == pacer_hr::ok);
            EXPECT(tokens[i] == 3 + i);
        }
        const std::uint64_t now = std::uint64_t(period) * (2 * latency + 2);
        for (std::uint32_t i = 0; i < latency; ++i) {  // `latency` presents fill the queue: nothing is replaced
            EXPECT(pacer.check_present_token(tokens[i]) == pacer_hr::ok);
            pacer.commit_present_token(tokens[i], 1);
        }
        EXPECT(pacer.frames_in_flight() == latency);
        pacer.commit_present_token(tokens[latency], 1);  // replaces the newest queued frame and takes its vblank
        EXPECT(pacer.frames_in_flight() == latency);
        wait = wait_async(pacer, infinite_timeout_ms);
        EXPECT(blocked(wait));
        clock.advance_us(period);
        EXPECT(finishes(wait, pacer));
        r = wait.get();
        EXPECT(r.hr == pacer_hr::ok && r.token == latency + 4);
        clock.advance_us(std::uint64_t(period) * (latency - 1));
        EXPECT(pacer.frames_in_flight() == 0);
        for (std::uint32_t i = 0; i + 1 < latency; ++i)  // the older frames keep their vblanks
            EXPECT(display_time(pacer, tokens[i]) == now + std::uint64_t(period) * (i + 1));
        EXPECT(display_time(pacer, tokens[latency - 1]) == no_deadline);  // replaced: never shown
        EXPECT(display_time(pacer, tokens[latency]) == now + std::uint64_t(period) * latency);

        // One present per wait, each a quarter period after its origin: nothing is replaced, each frame is shown at the
        // vblank after its origin, and that vblank is the next origin.
        token = r.token;
        clock.advance_us(period);  // back on an origin
        for (int frame = 0; frame < 8; ++frame) {
            const std::uint64_t origin = clock.now_us();
            std::uint64_t next = 0;
            EXPECT(pacer.wait_frame_event(frame_event_origin, 0, 0, &next) == pacer_hr::ok && next == token + 1);
            token = next;
            clock.advance_us(period / 4);
            pacer.commit_present_token(token, 1);
            EXPECT(pacer.frames_in_flight() == 1);
            clock.advance_us(period - period / 4);
            EXPECT(pacer.frames_in_flight() == 0);
            EXPECT(display_time(pacer, token) == origin + period);
        }
    }

    // SetFrameIntervalX lowering the latency below the frames already queued: the next present cuts the queue back
    // to the new latency from the newest end, and the oldest frame keeps its vblank.
    {
        ManualFrameClock clock;
        FramePacer pacer(clock);
        EXPECT(setup(pacer, 3));
        for (int i = 0; i < 3; ++i) pacer.note_untokened_present();  // vblanks 1, 2 and 3
        EXPECT(pacer.frames_in_flight() == 3);
        EXPECT(pacer.set_frame_interval(period, 1, 0) == pacer_hr::ok);
        EXPECT(pacer.frames_in_flight() == 3);  // already queued: left alone until the next present
        pacer.note_untokened_present();
        EXPECT(pacer.frames_in_flight() == 1);
        clock.advance_us(period - 1);
        EXPECT(pacer.frames_in_flight() == 1);
        clock.advance_us(1);  // vblank 1, the oldest frame's
        EXPECT(pacer.frames_in_flight() == 0);
    }

    // Tokens: only one that was never handed out is refused. Null, stale and repeated ones are accepted and reported
    // as such, so a title that presents with Token = 0, twice with one token, or out of order does not freeze.
    {
        using Issue = FramePacer::TokenIssue;
        ManualFrameClock clock;
        FramePacer pacer(clock);
        EXPECT(setup(pacer, 4));
        std::uint64_t t[3] = {};
        for (auto& token : t) {
            clock.advance_us(period);
            EXPECT(pacer.wait_frame_event(frame_event_origin, 0, 0, &token) == pacer_hr::ok);
        }
        EXPECT(t[0] == 1 && t[1] == 2 && t[2] == 3);
        Issue issue = Issue::stale;
        EXPECT(pacer.check_present_token(4, &issue) == pacer_hr::invalid_arg);     // never issued
        EXPECT(pacer.check_present_token(1000, &issue) == pacer_hr::invalid_arg);
        EXPECT(pacer.check_present_token(0, &issue) == pacer_hr::ok && issue == Issue::null_token);
        EXPECT(pacer.check_present_token(2, &issue) == pacer_hr::ok && issue == Issue::none);  // 1 is skipped
        pacer.commit_present_token(2, 1);
        EXPECT(pacer.check_present_token(2, &issue) == pacer_hr::ok && issue == Issue::repeated);
        EXPECT(pacer.check_present_token(1, &issue) == pacer_hr::ok && issue == Issue::stale);
        EXPECT(pacer.check_present_token(3, &issue) == pacer_hr::ok && issue == Issue::none);
        // Presenting a stale or null token does not move the last presented token back.
        pacer.commit_present_token(1, 1);
        pacer.commit_present_token(0, 1);
        EXPECT(pacer.check_present_token(2, &issue) == pacer_hr::ok && issue == Issue::repeated);
        EXPECT(pacer.check_present_token(3, nullptr) == pacer_hr::ok);
    }

    // Changing the interval while the origin event is armed and a waiter is blocked: the event stays armed, the grid
    // is re-based on the change, and the waiter comes back at the first origin of the new grid.
    {
        ManualFrameClock clock;
        FramePacer pacer(clock);
        EXPECT(setup(pacer, 1));
        auto waiter = wait_async(pacer, infinite_timeout_ms);
        EXPECT(blocked(waiter));
        clock.advance_us(period / 2);
        EXPECT(pacer.set_frame_interval(2 * period, 1, 0) == pacer_hr::ok);  // now = period / 2; new grid from here
        EXPECT(pacer.registered() && pacer.period_us() == 2 * period);
        clock.advance_us(period);  // the old grid's first origin has passed: still blocked
        EXPECT(blocked(waiter));
        clock.advance_us(period - 1);
        EXPECT(blocked(waiter));
        clock.advance_us(1);  // period / 2 + 2 * period: the first origin of the new grid
        Result r = waiter.get();
        EXPECT(r.hr == pacer_hr::ok && r.token == 1);
        std::uint32_t count = 1;
        FrameStatistic stat;
        EXPECT(pacer.get_frame_statistics(1, stat_frame_event, &count, &stat) == pacer_hr::ok);
        EXPECT(stat.time_us == period / 2 + 2 * period);
        // An offset larger than the new, shorter period is cut to fit.
        EXPECT(pacer.schedule_frame_event(frame_event_origin, period + 1000, 0) == pacer_hr::ok);
        EXPECT(pacer.set_frame_interval(period / 2, 1, 0) == pacer_hr::ok);
        EXPECT(pacer.registered());
    }

    // Scheduling inside a period, before the offset has passed: that period's origin is still to come.
    {
        ManualFrameClock clock;
        FramePacer pacer(clock);
        EXPECT(pacer.set_frame_interval(period, 1, 0) == pacer_hr::ok);
        clock.advance_us(period * 2 + 2000);
        EXPECT(pacer.schedule_frame_event(frame_event_origin, 5000, 0) == pacer_hr::ok);
        auto wait = wait_async(pacer, infinite_timeout_ms);
        EXPECT(blocked(wait));
        clock.advance_us(2999);
        EXPECT(blocked(wait));
        clock.advance_us(1);  // 2 * period + 5000
        EXPECT(wait.get().token == 1);
        std::uint32_t count = 1;
        FrameStatistic stat;
        EXPECT(pacer.get_frame_statistics(1, stat_frame_event, &count, &stat) == pacer_hr::ok);
        EXPECT(stat.time_us == period * 2 + 5000);
    }

    // Statistics: unknown tokens and types the pacer cannot give report S_FALSE and count 0.
    {
        ManualFrameClock clock;
        FramePacer pacer(clock);
        EXPECT(setup(pacer, 1));
        clock.advance_us(period);
        std::uint64_t token = 0;
        EXPECT(pacer.wait_frame_event(frame_event_origin, 0, 0, &token) == pacer_hr::ok);
        std::uint32_t count = 4;
        FrameStatistic stats[4];
        EXPECT(pacer.get_frame_statistics(77, stat_frame_event, &count, stats) == pacer_hr::false_ && count == 0);
        count = 4;
        EXPECT(pacer.get_frame_statistics(token, stat_render | stat_frame_interval, &count, stats) == pacer_hr::false_ &&
               count == 0);
        // Not presented yet: the event is there, present and display are not.
        count = 4;
        EXPECT(pacer.get_frame_statistics(token, stat_frame_event | stat_present | stat_display, &count, stats) ==
                   pacer_hr::false_ && count == 1 && stats[0].type == stat_frame_event);
        count = 0;
        EXPECT(pacer.get_frame_statistics(token, stat_frame_event, nullptr, stats) == pacer_hr::invalid_arg);
        // A null array only counts.
        count = 4;
        EXPECT(pacer.get_frame_statistics(token, stat_frame_event, &count, nullptr) == pacer_hr::ok && count == 1);
        // Presented but not shown: display missing. Then shown.
        pacer.commit_present_token(token, 1);
        count = 4;
        EXPECT(pacer.get_frame_statistics(token, stat_present | stat_display, &count, stats) == pacer_hr::false_ &&
               count == 1 && stats[0].type == stat_present);
        clock.advance_us(period);
        count = 4;
        EXPECT(pacer.get_frame_statistics(token, stat_present | stat_display, &count, stats) == pacer_hr::ok &&
               count == 2);
        // Too little room: written up to the room, S_FALSE.
        count = 1;
        EXPECT(pacer.get_frame_statistics(token, stat_present | stat_display, &count, stats) == pacer_hr::false_ &&
               count == 1);
    }

    // Suspend: a blocked waiter fails, presents are refused, registration is gone after resume and must be redone;
    // tokens keep counting.
    {
        ManualFrameClock clock;
        FramePacer pacer(clock);
        EXPECT(setup(pacer, 1));
        clock.advance_us(period);
        std::uint64_t token = 0;
        EXPECT(pacer.wait_frame_event(frame_event_origin, 0, 0, &token) == pacer_hr::ok && token == 1);
        auto waiter = wait_async(pacer, infinite_timeout_ms);
        EXPECT(blocked(waiter));
        pacer.suspend();
        EXPECT(waiter.get().hr == pacer_hr::invalid_state);
        EXPECT(pacer.suspended());
        EXPECT(pacer.check_present_token(1) == pacer_hr::invalid_state);
        EXPECT(pacer.set_frame_interval(period, 1, 0) == pacer_hr::invalid_state);
        EXPECT(pacer.wait_frame_event(frame_event_origin, 0, 0, &token) == pacer_hr::invalid_state);
        pacer.resume();
        EXPECT(!pacer.suspended() && !pacer.registered());
        EXPECT(pacer.wait_frame_event(frame_event_origin, 0, 0, &token) == pacer_hr::invalid_state);  // not re-registered
        EXPECT(pacer.schedule_frame_event(frame_event_origin, 0, 0) == pacer_hr::invalid_state);       // interval too
        EXPECT(setup(pacer, 1));
        clock.advance_us(period);
        EXPECT(pacer.wait_frame_event(frame_event_origin, 0, 0, &token) == pacer_hr::ok && token == 2);
        EXPECT(pacer.check_present_token(2) == pacer_hr::ok);
        EXPECT(pacer.check_present_token(1) == pacer_hr::ok);  // never presented, so not stale
    }

    // Destroying a pacer with a blocked waiter lets the waiter go.
    {
        ManualFrameClock clock;
        std::future<Result> waiter;
        {
            FramePacer pacer(clock);
            EXPECT(setup(pacer, 1));
            waiter = wait_async(pacer, infinite_timeout_ms);
            EXPECT(blocked(waiter));
        }
        EXPECT(waiter.get().hr == pacer_hr::invalid_state);
    }

    // ImmediateThresholdPercent: the pure decision (late_by, interval, threshold -> mode).
    {
        const auto sync = [](std::uint64_t late, std::uint32_t interval, float threshold) {
            return decide_present_sync(late, interval, threshold);
        };
        constexpr PresentSync v = PresentSync::vblank, i = PresentSync::immediate;
        const float nan = std::numeric_limits<float>::quiet_NaN();
        EXPECT(sync(0, period, 100.0f) == v);                  // on time: nothing to tear
        EXPECT(sync(100, period, 0.0f) == v);                  // threshold 0 disables tearing
        EXPECT(sync(100, period, -5.0f) == v);
        EXPECT(sync(100, period, nan) == v);
        EXPECT(sync(100, 0, 100.0f) == v);                     // unpaced
        EXPECT(sync(1, period, 1.0f) == i);                    // 0.00625% scanned, within 1%
        EXPECT(sync(159, period, 1.0f) == i);                  // 0.99375%
        EXPECT(sync(160, period, 1.0f) == i);                  // exactly 1%: the threshold itself still tears
        EXPECT(sync(161, period, 1.0f) == v);                  // just past it
        EXPECT(sync(8000, period, 49.0f) == v);                // 50% scanned
        EXPECT(sync(8000, period, 50.0f) == i);
        EXPECT(sync(8000, period, 51.0f) == i);
        EXPECT(sync(15999, period, 100.0f) == i);              // 100: anywhere in the scan
        EXPECT(sync(15999, period, 250.0f) == i);              // above 100 is clamped
        EXPECT(sync(15999, period, 99.0f) == v);
        EXPECT(sync(period + 4000, period, 25.0f) == i);       // a whole interval and a quarter late: the scan is at 25%
        EXPECT(sync(period + 4000, period, 24.0f) == v);
        EXPECT(sync(3 * period, period, 10.0f) == i);          // late by whole intervals: the scan is at the top
        EXPECT(scan_percent(8000, period) == 50.0f);
        EXPECT(scan_percent(0, period) == 0.0f && scan_percent(5, 0) == 0.0f);
    }

    // The same through the pacer: lateness against the vblank the token was due at, and what a tear does to the frame.
    {
        ManualFrameClock clock;
        FramePacer pacer(clock);
        EXPECT(setup(pacer, 2));
        clock.advance_us(period);  // origin 1; its frame is due at vblank 2 (t = 2 * period)
        std::uint64_t token1 = 0, token2 = 0;
        EXPECT(pacer.wait_frame_event(frame_event_origin, 0, 0, &token1) == pacer_hr::ok);
        FramePacer::PresentTiming timing = pacer.present_timing(token1);
        EXPECT(timing.period_us == period && timing.late_us == 0);  // before the vblank it is due at
        EXPECT(pacer.present_timing(0).late_us == 0 && pacer.present_timing(99).late_us == 0);  // null, unknown
        clock.advance_us(period);  // exactly at its vblank: not late
        EXPECT(pacer.present_timing(token1).late_us == 0);
        clock.advance_us(8000);    // half an interval late
        timing = pacer.present_timing(token1);
        EXPECT(timing.late_us == 8000);
        const PresentSync mode = decide_present_sync(timing.late_us, timing.period_us, 60.0f);
        EXPECT(mode == PresentSync::immediate);
        pacer.commit_present_token(token1, 1, mode);
        EXPECT(pacer.immediate_presents() == 1);
        EXPECT(pacer.present_timing(token1).late_us == 0);  // presented already: a repeated token never tears
        EXPECT(pacer.frames_in_flight() == 0);  // shown at once, not queued for the next vblank
        std::uint32_t count = 4;
        FrameStatistic stats[4];
        EXPECT(pacer.get_frame_statistics(token1, stat_present | stat_display, &count, stats) == pacer_hr::ok && count == 2);
        EXPECT(stats[0].type == stat_present && stats[1].type == stat_display);
        EXPECT(stats[1].time_us == stats[0].time_us && stats[1].time_us == 2 * period + 8000);
        EXPECT(stats[1].percent_scanned == 50.0f && stats[0].percent_scanned == 0.0f);

        // The next frame, late by the same amount but with a threshold that is too small: waits for its vblank.
        clock.advance_us(period - 8000);  // t = 3 * period: origin 3
        EXPECT(pacer.wait_frame_event(frame_event_origin, 0, 0, &token2) == pacer_hr::ok);
        clock.advance_us(period + 8000);  // due at 4 * period, now 4.5
        timing = pacer.present_timing(token2);
        EXPECT(timing.late_us == 8000);
        EXPECT(decide_present_sync(timing.late_us, timing.period_us, 40.0f) == PresentSync::vblank);
        pacer.commit_present_token(token2, 1, PresentSync::vblank);
        EXPECT(pacer.immediate_presents() == 1 && pacer.frames_in_flight() == 1);
        EXPECT(pacer.get_frame_statistics(token2, stat_display, &count, stats) == pacer_hr::false_);  // not shown yet
        clock.advance_us(period);  // vblank at 5 * period
        count = 4;
        EXPECT(pacer.get_frame_statistics(token2, stat_display, &count, stats) == pacer_hr::ok && count == 1);
        EXPECT(stats[0].time_us == 5 * period && stats[0].percent_scanned == 0.0f);
    }

    // Rescheduling changes future origins but not the vblank an issued token was due at.
    {
        ManualFrameClock clock;
        FramePacer pacer(clock);
        EXPECT(setup(pacer, 1, period / 4));
        clock.advance_us(period + period / 4);
        std::uint64_t token = 0;
        EXPECT(pacer.wait_frame_event(frame_event_origin, 0, 0, &token) == pacer_hr::ok);
        EXPECT(pacer.schedule_frame_event(frame_event_origin, period / 2, 0) == pacer_hr::ok);

        clock.advance_us(3 * period / 4);  // vblank 2, still the deadline of the original token
        EXPECT(pacer.present_timing(token).late_us == 0);
        clock.advance_us(period / 4);
        const auto timing = pacer.present_timing(token);
        EXPECT(timing.late_us == period / 4);
        const auto sync = decide_present_sync(timing.late_us, timing.period_us, 30.0f);
        EXPECT(sync == PresentSync::immediate);
        pacer.commit_present_token(token, 1, sync);

        FrameStatistic display{};
        std::uint32_t count = 1;
        EXPECT(pacer.get_frame_statistics(token, stat_display, &count, &display) == pacer_hr::ok);
        EXPECT(count == 1 && display.percent_scanned == 25.0f);
    }

    // A frame behind one that still waits for its vblank cannot be shown at once (it would overtake it): not late.
    {
        ManualFrameClock clock;
        FramePacer pacer(clock);
        EXPECT(setup(pacer, 2));
        std::uint64_t token1 = 0, token2 = 0;
        clock.advance_us(period);      // origin 1
        EXPECT(pacer.wait_frame_event(frame_event_origin, 0, 0, &token1) == pacer_hr::ok);
        clock.advance_us(period);      // origin 2 (nothing is in flight yet, so the second token is handed out)
        EXPECT(pacer.wait_frame_event(frame_event_origin, 0, 0, &token2) == pacer_hr::ok);
        clock.advance_us(period + 1600);  // 3.1 periods: token1 is presented, a whole interval late, queued for vblank 4
        pacer.commit_present_token(token1, 1);
        EXPECT(pacer.frames_in_flight() == 1);
        clock.advance_us(1600);        // 3.2: token2 was due at vblank 3 and is late, but token1 is still queued
        EXPECT(pacer.present_timing(token2).late_us == 0 && pacer.present_timing(token2).period_us == period);
        clock.advance_us(period);      // 4.2: token1 was shown at vblank 4; token2 is late and nothing is queued
        EXPECT(pacer.present_timing(token2).late_us == 19200);  // 4.2 periods now, due at 3
        EXPECT(pacer.immediate_presents() == 0);
    }

    // Presents made after SetFrameIntervalX but before the origin event is scheduled are unpaced (PresentX does not
    // hold them for a vblank), so they are shown at once and are not frames in flight: however many there were, the
    // first wait returns at the first origin.
    {
        ManualFrameClock clock;
        FramePacer pacer(clock);
        EXPECT(pacer.set_frame_interval(period, 1, 0) == pacer_hr::ok);
        EXPECT(!pacer.registered());
        for (int i = 0; i < 1000; ++i) pacer.note_untokened_present();  // blank presents of a loading screen
        pacer.commit_present_token(0, 1);                               // and one with the null token
        EXPECT(pacer.frames_in_flight() == 0);
        EXPECT(pacer.schedule_frame_event(frame_event_origin, 0, 0) == pacer_hr::ok);
        auto wait = wait_async(pacer, infinite_timeout_ms);
        EXPECT(blocked(wait));
        clock.advance_us(period);
        if (wait.wait_for(std::chrono::seconds(10)) != std::future_status::ready) {
            pacer.suspend();  // lets the waiter go, so that the failure below ends the test and does not hang it
            EXPECT(!"the first wait returns one period after the event is armed");
        }
        Result r = wait.get();
        EXPECT(r.hr == pacer_hr::ok && r.token == 1);
        // Once the event is armed, a present is a frame in flight again.
        pacer.note_untokened_present();
        EXPECT(pacer.frames_in_flight() == 1);
    }

    // Uncapped (LEONIDA_PRESENT_UNCAP): a wait hands a token out at once instead of at the next interval origin,
    // and a presented frame counts as shown at once. Registration and suspension keep their errors.
    {
        ManualFrameClock clock;
        FramePacer pacer(clock);
        pacer.set_uncapped(true);
        EXPECT(pacer.uncapped());
        std::uint64_t token = 99;
        // Still invalid_state before registration, after suspend, and after resume until re-registered.
        EXPECT(pacer.wait_frame_event(frame_event_origin, infinite_timeout_ms, 0, &token) == pacer_hr::invalid_state &&
               token == 99);
        EXPECT(setup(pacer, 1));
        // Tokens at once, no clock advance; as many in a row as asked (the latency limit does not throttle),
        // each from a wait that would still be blocked without the flag.
        for (std::uint64_t i = 1; i <= 3; ++i) {
            auto wait = wait_async(pacer, infinite_timeout_ms);
            Result r = wait.get();
            EXPECT(r.hr == pacer_hr::ok && r.token == i);
        }
        pacer.commit_present_token(1, 1);
        pacer.commit_present_token(2, 1);
        pacer.commit_present_token(3, 1);
        EXPECT(pacer.frames_in_flight() == 0);  // nothing waits for a vblank
        std::uint32_t count = 2;
        FrameStatistic stats[2];
        EXPECT(pacer.get_frame_statistics(3, stat_present | stat_display, &count, stats) == pacer_hr::ok);
        EXPECT(count == 2 && stats[0].type == stat_present && stats[0].time_us == 0 &&
               stats[1].type == stat_display && stats[1].time_us == 0);
        // An untokened (blank) present is not queued on the grid either.
        pacer.note_untokened_present();
        EXPECT(pacer.frames_in_flight() == 0);
        // An uncapped frame is due at once: present_timing reports no lateness.
        const FramePacer::PresentTiming timing = pacer.present_timing(3);
        EXPECT(timing.late_us == 0 && timing.period_us == 0);
        pacer.suspend();
        EXPECT(pacer.wait_frame_event(frame_event_origin, 0, 0, &token) == pacer_hr::invalid_state);
        pacer.resume();
        EXPECT(pacer.wait_frame_event(frame_event_origin, 0, 0, &token) == pacer_hr::invalid_state);
        EXPECT(setup(pacer, 1));
        clock.advance_us(1234);
        EXPECT(pacer.wait_frame_event(frame_event_origin, 0, 0, &token) == pacer_hr::ok && token == 4);
        // The flag survives suspend/resume (set once, by the environment).
        pacer.suspend();
        pacer.resume();
        EXPECT(pacer.uncapped());
        EXPECT(setup(pacer, 1));
        EXPECT(pacer.wait_frame_event(frame_event_origin, 0, 0, &token) == pacer_hr::ok && token == 5);
        // Turning the cap back on restores the grid: the next wait blocks until the next origin (the grid was
        // re-based at the last SetFrameIntervalX, so a whole period is the next origin).
        pacer.set_uncapped(false);
        auto wait = wait_async(pacer, infinite_timeout_ms);
        EXPECT(blocked(wait));
        clock.advance_us(period);
        EXPECT(wait.get().token == 6);
    }

    // Real clock: five 10 ms vblanks take about 50 ms.
    {
        SteadyFrameClock clock;
        FramePacer pacer(clock);
        EXPECT(pacer.set_frame_interval(10000, 1, 0) == pacer_hr::ok);
        EXPECT(pacer.schedule_frame_event(frame_event_origin, 0, 0) == pacer_hr::ok);
        const auto start = std::chrono::steady_clock::now();
        std::uint64_t token = 0, last = 0;
        for (int i = 0; i < 5; ++i) {
            EXPECT(pacer.wait_frame_event(frame_event_origin, 2000, 0, &token) == pacer_hr::ok);
            EXPECT(token == last + 1);
            last = token;
        }
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
        std::printf("real clock: 5 origins at 10 ms took %lld ms\n", static_cast<long long>(ms.count()));
        EXPECT(ms.count() >= 45 && ms.count() < 1000);
    }

    std::printf("frame_pacer_test: all cases pass\n");
    return 0;
}
