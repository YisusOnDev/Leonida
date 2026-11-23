// CPU-only regression for D3D-496 with unpinned Waits: a fence whose last reference goes inside a release.
//
// Queue A holds Signal(F, 1) behind Wait(G, 1); queue B holds a batch behind Wait(F, 2), a value only F's postponed
// CPU Signal(2) brings. A Wait does not keep F alive, so when the CPU signals G and A's Signal is released, the pin A
// drops after its lock can be F's last reference: F is removed on the releasing thread, inside pump_ready, and its
// postponed Signal(2) is cancelled with it. remove_fence has to take F off B's held Wait there too. Once F is erased,
// pending() counts that Wait as met, and B would be submitted still waiting on F, which never reaches 2: B's queue
// then stalls for good (d3d12_fence_timing_test under load: "B's batch ran 49/200", and nothing after).
#include "../src/queue_holds.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <mutex>
#include <utility>
#include <vector>

namespace {

using Holds = leonida::QueueHolds<int>;
constexpr int kGate = 1;
constexpr int kFence = 2;

struct Op {
    std::vector<int> waits;
    std::vector<std::uint64_t> values;
    std::uint64_t seq = 0;
};

// A queue as VulkanQueue holds: operations released in order once their waits are met, and what it drops after
// releasing (the pins) dropped after its own lock.
class FakeQueue final : public Holds::Holder {
public:
    FakeQueue(Holds& holds, const std::vector<int>& removed) : holds_(holds), removed_(removed) {}

    void hold(std::vector<int> waits, std::vector<std::uint64_t> values) {
        std::lock_guard lock(mutex_);
        held_.push_back({std::move(waits), std::move(values), holds_.next_seq()});
        holds_.held_changed(1);
    }

    std::function<void()> after_release;  // run once, after the lock, when something was released
    int released = 0;
    int stale = 0;  // released still waiting on a removed semaphore: on the GPU that wait is never met

    bool release_ready() override {
        bool any = false;
        {
            std::lock_guard lock(mutex_);
            while (!held_.empty() && ready(held_.front())) {
                for (int wait : held_.front().waits)
                    if (std::find(removed_.begin(), removed_.end(), wait) != removed_.end()) ++stale;
                held_.erase(held_.begin());
                holds_.held_changed(-1);
                ++released;
                any = true;
            }
        }
        if (any && after_release) {
            std::function<void()> fn = std::move(after_release);
            after_release = nullptr;
            fn();
        }
        return any;
    }

    // Takes the queue's lock: called with it held, this would deadlock (the test then times out).
    void discard_waiting_on(int semaphore) override {
        std::lock_guard lock(mutex_);
        for (Op& op : held_)
            for (std::size_t i = 0; i < op.waits.size();) {
                if (op.waits[i] != semaphore) {
                    ++i;
                    continue;
                }
                op.waits.erase(op.waits.begin() + std::ptrdiff_t(i));
                op.values.erase(op.values.begin() + std::ptrdiff_t(i));
            }
    }

    std::uint64_t oldest_held_seq() override {
        std::lock_guard lock(mutex_);
        return held_.empty() ? 0 : held_.front().seq;
    }

private:
    bool ready(const Op& op) const {
        for (std::size_t i = 0; i < op.waits.size(); ++i)
            if (holds_.pending(op.waits[i], op.values[i])) return false;
        return true;
    }

    Holds& holds_;
    const std::vector<int>& removed_;
    std::mutex mutex_;
    std::vector<Op> held_;
};

// F released inside A's release (the pin A drops) or by the application outside any release, with A attached first
// or second (so B's turn in the pass comes after or before A's). 0 when A and B each ran once, B not waiting on F.
int check(bool inside_release, bool a_first) {
    Holds holds;
    std::vector<int> removed;
    FakeQueue a(holds, removed), b(holds, removed);
    holds.attach(a_first ? static_cast<Holds::Holder*>(&a) : &b);
    holds.attach(a_first ? static_cast<Holds::Holder*>(&b) : &a);
    holds.add_fence(kGate, 0);
    holds.add_fence(kFence, 0);
    a.hold({kGate}, {1});   // Wait(G, 1), then Signal(F, 1)
    b.hold({kFence}, {2});  // Wait(F, 2), then the batch; only F's postponed CPU Signal(2) reaches 2
    const auto release_fence = [&] {
        removed.push_back(kFence);  // F's last reference: its postponed Signal(2) is cancelled with it
        holds.remove_fence(kFence);
    };
    if (inside_release) {
        a.after_release = [&] {
            holds.signaled(kFence, 1);  // A's Signal(F, 1) was submitted
            release_fence();            // and the pin A drops after its lock was F's last reference
        };
        holds.signaled(kGate, 1);  // the CPU opens the gate
        holds.pump_ready();
    } else {
        holds.signaled(kGate, 1);
        holds.pump_ready();  // A runs; the application still holds F
        holds.signaled(kFence, 1);
        release_fence();
    }
    const bool ok = a.released == 1 && b.released == 1 && b.stale == 0 && !holds.any_held();
    std::printf("%s, A attached %s: A ran %d, B ran %d, B submitted waiting on the released fence %d: %s\n",
                inside_release ? "released inside A's release" : "released outside a release",
                a_first ? "first" : "second", a.released, b.released, b.stale, ok ? "ok" : "FAIL");
    holds.detach(&a);
    holds.detach(&b);
    return ok ? 0 : 1;
}

}  // namespace

int main() {
    int failures = 0;
    for (const bool inside_release : {true, false})
        for (const bool a_first : {true, false}) failures += check(inside_release, a_first);
    if (failures) {
        std::fprintf(stderr, "%d case(s) failed\n", failures);
        return 1;
    }
    std::puts("a fence released inside a release is forgotten by every queue before it is erased");
    return 0;
}
