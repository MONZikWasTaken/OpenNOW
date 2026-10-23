#pragma once

#include <memory>

class QRhi;
class QRhiCommandBuffer;
class QRhiTexture;

class StreamFrameInterpolator
{
public:
    StreamFrameInterpolator();
    ~StreamFrameInterpolator();
    StreamFrameInterpolator(const StreamFrameInterpolator &) = delete;
    StreamFrameInterpolator &operator=(const StreamFrameInterpolator &) = delete;

    // This engine synthesises the t = 0.5 midpoint only; higher factors fall back to it.
    static constexpr int MaximumFactor = 2;

    bool initialize(QRhi *rhi);
    bool ingest(QRhiCommandBuffer *cb, QRhiTexture *source, int factor = 2);
    QRhiTexture *currentTexture() const;
    QRhiTexture *midpointTexture() const;
    // Generated frames available for the current pair (1-based index between the two
    // source frames, in presentation order).
    int generatedCount() const;
    QRhiTexture *generatedTexture(int index) const;
    bool hasPair() const;
    void reset();
    void release();

private:
    struct State;
    std::unique_ptr<State> m_state;
};
