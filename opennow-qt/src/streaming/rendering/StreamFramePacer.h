#pragma once

#include <cmath>
#include <cstdint>
#include <algorithm>
#include <array>

// Schedules locally generated frames between consecutive source frames.
//
// A generation factor N presents N-1 generated frames per source interval, evenly
// spaced: generated frame k (1-based) is due at arrival + (k-1) * interval / N and the
// original frame at arrival + (N-1) * interval / N. The first generated frame is presented
// as soon as it exists, which is why its slot starts at the arrival time. Factor 2 is the
// classic midpoint schedule.
class StreamFramePacer
{
public:
    enum class Result { Duplicate, WarmingUp, Interpolate, Discontinuity, DisplayTooSlow, Overloaded, SourceRateLimit };
    enum class TimingSource { None, SourceTimestamps, ArrivalCadence };
    enum class Rejection { None, SequenceGap, TimestampRegression, TimestampJump, ArrivalGap, CadenceUnavailable };

    struct Next {
        enum class Kind { None, Generated, Original };
        Kind kind = Kind::None;
        int index = 0;   // 1-based generated frame index when kind == Generated
    };

    static constexpr int MinimumFactor = 2;
    static constexpr int MaximumFactor = 4;
    // Below this source interval (250 FPS) generation is pointless and the pacing
    // noise exceeds the slot width.
    static constexpr std::uint64_t MinimumSourceInterval = 4'000'000;

    Result source(std::uint64_t sequence, std::uint64_t timestamp,
                  std::int64_t now, double refreshRate, int factor = 2)
    {
        m_factor = std::clamp(factor, MinimumFactor, MaximumFactor);
        if (m_hasSource && sequence == m_sequence && timestamp == m_timestamp)
            return Result::Duplicate;
        m_sequenceDelta = sequence >= m_sequence ? sequence - m_sequence : 0;
        // The presenter acquires the latest decoded frame, so a decoder or network burst
        // skips one source frame. The pair around it is still adjacent in time and the
        // generated frame stands in for the one that was never shown; wider gaps, or a
        // skip before the cadence is known, re-warm.
        const std::uint64_t span = m_interval && m_sequenceDelta == 2 ? 2 : 1;
        const bool consecutive = m_sequenceDelta == 1 || span == 2;
        const bool rewound = timestamp && m_timestamp && timestamp < m_timestamp;
        m_timestampDelta = timestamp >= m_timestamp ? timestamp - m_timestamp : 0;
        m_arrivalDelta = now >= m_arrival ? now - m_arrival : 0;
        const bool jumped = m_timestamp && timestamp && m_timestampDelta > 250'000'000;
        const auto gapLimit = m_interval ? std::max<std::int64_t>(50'000'000, m_interval * 2) : 75'000'000;
        const auto arrivalPerFrame = m_arrivalDelta / std::int64_t(span);
        const auto timestampPerFrame = m_timestampDelta / span;
        const bool pending = m_pending;
        const bool first = !m_hasSource;
        m_hasSource = true;
        m_sequence = sequence;
        m_timestamp = timestamp;
        m_arrival = now;
        m_pending = false;
        m_pendingCount = 0;
        m_pendingIndex = 0;
        m_rejection = Rejection::None;
        m_timingSource = TimingSource::None;
        if (first) {
            m_sequenceDelta = 0;
            m_timestampDelta = 0;
            m_arrivalDelta = 0;
            return Result::WarmingUp;
        }
        const auto reject = [this](Rejection reason) {
            m_rejection = reason;
            m_interval = 0;
            m_arrivalCount = 0;
            m_arrivalSlot = 0;
            return Result::Discontinuity;
        };
        if (!consecutive) return reject(Rejection::SequenceGap);
        if (rewound) return reject(Rejection::TimestampRegression);
        if (jumped) return reject(Rejection::TimestampJump);
        if (arrivalPerFrame > gapLimit) return reject(Rejection::ArrivalGap);
        m_arrivals[m_arrivalSlot] = arrivalPerFrame;
        m_arrivalSlot = (m_arrivalSlot + 1) % m_arrivals.size();
        m_arrivalCount = std::min(m_arrivalCount + 1, m_arrivals.size());
        auto sorted = m_arrivals;
        std::sort(sorted.begin(), sorted.begin() + m_arrivalCount);
        const auto observed = m_arrivalCount
            ? (sorted[(m_arrivalCount - 1) / 2] + sorted[m_arrivalCount / 2]) / 2 : 0;
        const bool arrivalUsable = observed >= 2'000'000 && observed <= 40'000'000;
        const bool sourceUsable = timestampPerFrame >= 2'000'000 && timestampPerFrame <= 40'000'000
            && (!arrivalUsable || (timestampPerFrame * 4 >= std::uint64_t(observed) * 3
                                  && timestampPerFrame * 4 <= std::uint64_t(observed) * 5));
        if (sourceUsable) {
            m_interval = timestampPerFrame;
            m_timingSource = TimingSource::SourceTimestamps;
        } else if (arrivalUsable) {
            m_interval = observed;
            m_timingSource = TimingSource::ArrivalCadence;
        } else {
            m_interval = 0;
            m_rejection = Rejection::CadenceUnavailable;
            return Result::Discontinuity;
        }
        if (m_interval < MinimumSourceInterval) return Result::SourceRateLimit;
        // The display must show factor frames per source interval; 5 % tolerance for
        // 59.94 Hz class sources on 120 Hz class displays.
        if (!std::isfinite(refreshRate)
            || refreshRate < (double(m_factor) - 0.05) * 1e9 / double(m_interval))
            return Result::DisplayTooSlow;
        if (pending) {
            // A source frame that lands before the previous pair finished presenting drops
            // that pair's remaining schedule. Isolated collisions are decoder or network
            // bursts; four within a second mean the presenter cannot keep up, so generation
            // pauses for two seconds instead of queueing.
            m_collisions[m_collisionSlot] = now;
            m_collisionSlot = (m_collisionSlot + 1) % m_collisions.size();
            m_collisionCount = std::min(m_collisionCount + 1, m_collisions.size());
            if (m_collisionCount == m_collisions.size()
                && now - *std::min_element(m_collisions.begin(), m_collisions.end()) <= 1'000'000'000)
                m_cooldownUntil = now + 2'000'000'000;
        }
        if (now < m_cooldownUntil) return Result::Overloaded;
        return Result::Interpolate;
    }

    // `count` generated frames exist for the pair that ended at `now`; frame 1 is being
    // presented now, the rest and the original follow at interval / (count + 1) slots.
    void generated(std::int64_t now, int count)
    {
        count = std::clamp(count, 0, MaximumFactor - 1);
        if (count == 0 || !m_interval) {
            m_pending = false;
            m_pendingCount = 0;
            m_pendingIndex = 0;
            return;
        }
        m_pending = true;
        m_pendingCount = count;
        m_pendingIndex = 1;
        m_slot = std::int64_t(m_interval / std::uint64_t(count + 1));
        m_nextDue = now + m_slot;
    }

    // Advances the schedule once the next slot is within one and a half refresh periods.
    Next takeNext(std::int64_t now, double refreshRate)
    {
        if (!m_pending) return {};
        const auto lead = refreshRate > 0 ? std::int64_t(1.5e9 / refreshRate) : 0;
        if (now + lead < m_nextDue) return {};
        ++m_pendingIndex;
        if (m_pendingIndex <= m_pendingCount) {
            m_nextDue += m_slot;
            return {Next::Kind::Generated, m_pendingIndex};
        }
        m_pending = false;
        m_pendingCount = 0;
        m_pendingIndex = 0;
        return {Next::Kind::Original, 0};
    }

    // Factor-2 conveniences kept for callers that only know about midpoints.
    void midpoint(std::int64_t now) { generated(now, 1); }
    bool takeOriginal(std::int64_t now, double refreshRate)
    {
        return takeNext(now, refreshRate).kind == Next::Kind::Original;
    }

    bool pending() const { return m_pending; }
    int factor() const { return m_factor; }
    int pendingCount() const { return m_pendingCount; }
    int pendingIndex() const { return m_pendingIndex; }
    TimingSource timingSource() const { return m_timingSource; }
    Rejection rejection() const { return m_rejection; }
    std::uint64_t interval() const { return m_interval; }
    std::uint64_t timestampDelta() const { return m_timestampDelta; }
    std::int64_t arrivalDelta() const { return m_arrivalDelta; }
    std::uint64_t sequenceDelta() const { return m_sequenceDelta; }
    void reset() { *this = {}; }

private:
    std::uint64_t m_sequence = 0;
    std::uint64_t m_timestamp = 0;
    std::uint64_t m_interval = 0;
    std::uint64_t m_timestampDelta = 0;
    std::uint64_t m_sequenceDelta = 0;
    std::int64_t m_arrivalDelta = 0;
    std::array<std::int64_t, 8> m_arrivals{};
    std::size_t m_arrivalCount = 0;
    std::size_t m_arrivalSlot = 0;
    TimingSource m_timingSource = TimingSource::None;
    Rejection m_rejection = Rejection::None;
    std::int64_t m_arrival = 0;
    std::int64_t m_nextDue = 0;
    std::int64_t m_slot = 0;
    std::int64_t m_cooldownUntil = 0;
    std::array<std::int64_t, 4> m_collisions{};
    std::size_t m_collisionSlot = 0;
    std::size_t m_collisionCount = 0;
    int m_factor = 2;
    int m_pendingCount = 0;
    int m_pendingIndex = 0;
    bool m_hasSource = false;
    bool m_pending = false;
};
