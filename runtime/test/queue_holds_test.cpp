// CPU-only regression for the releases QueueHolds defers while queues hold work (D3D-203, docs section 4.4): a
// release asked for with after_held() waits for every operation held when it was asked for, on any queue, whatever
// order those operations are released or dropped in, and for nothing held after it; it runs once.
#include "../src/queue_holds.h"

#include <cstdint>
#include <cstdio>
#include <deque>
#include <utility>

namespace {

using Holds = leonida::QueueHolds<int>;

// A queue that holds operations in order and releases its front ones once they are marked ready.
struct FakeQueue final : Holds::Holder {
    explicit FakeQueue(Holds& holds) : holds(holds) { holds.attach(this); }
    ~FakeQueue() { holds.detach(this); }

    std::uint64_t hold() {
        const std::uint64_t seq = holds.next_seq();
        ops.push_back({seq, false});
        holds.held_changed(1);
        return seq;
    }
    // The operation's wait is satisfied: the queue releases what is ready at its front.
    void make_ready(std::uint64_t seq) {
        for (auto& op : ops)
            if (op.first == seq) op.second = true;
        holds.pump_ready();
    }
    // The operation is dropped (its fence released or its queue destroyed).
    void drop(std::uint64_t seq) {
        for (auto it = ops.begin(); it != ops.end(); ++it)
            if (it->first == seq) {
                ops.erase(it);
                holds.held_changed(-1);
                break;
            }
        holds.pump_ready();
    }

    bool release_ready() override {
        bool released = false;
        while (!ops.empty() && ops.front().second) {
            ops.pop_front();
            holds.held_changed(-1);
            released = true;
        }
        return released;
    }
    void discard_waiting_on(int) override {}
    std::uint64_t oldest_held_seq() override { return ops.empty() ? 0 : ops.front().first; }

    Holds& holds;
    std::deque<std::pair<std::uint64_t, bool>> ops;  // sequence number, ready
};

}  // namespace

int main() {
    Holds holds;
    const int owner = 0;  // the device whose resources are released
    int released = 0;
    const auto release = [&] { ++released; };
    FakeQueue a(holds), b(holds);

    // Nothing held: the release runs at once.
    holds.after_held(&owner, release);
    if (released != 1 || holds.pending_after(&owner) != 0) return 1;

    // Queue A holds two operations (a batch naming the resource, then its Signal), queue B one; the resource is
    // released; B then holds another operation, which cannot name it.
    const auto a1 = a.hold();
    const auto a2 = a.hold();
    const auto b1 = b.hold();
    holds.after_held(&owner, release);
    if (released != 1 || holds.pending_after(&owner) != 1) return 2;
    const auto b2 = b.hold();

    // B's operation is released first, then A's batch: A's Signal is still held, so the release waits.
    b.make_ready(b1);
    if (released != 1) return 3;
    a.make_ready(a1);
    if (released != 1) return 4;
    // A's last operation is released: so is the resource, once.
    a.make_ready(a2);
    if (released != 2 || holds.pending_after(&owner) != 0) return 5;
    holds.pump_ready();
    if (released != 2) return 6;

    // A release that waits for b2 alone, then one behind a new hold that waits for both; c1 goes first.
    holds.after_held(&owner, release);
    const auto c1 = a.hold();
    holds.after_held(&owner, release);
    if (released != 2 || holds.pending_after(&owner) != 2) return 7;
    a.make_ready(c1);
    if (released != 2) return 8;
    // b2 is dropped: the same as released for what waits on it.
    b.drop(b2);
    if (released != 4 || holds.pending_after(&owner) != 0 || holds.any_held()) return 9;

    std::puts("releases waited for the operations held before them, in any completion order, and ran once");
    return 0;
}
