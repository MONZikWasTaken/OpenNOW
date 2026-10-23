#include "streaming/rendering/FrameGenerationBenchmark.h"

#include <QTest>
#include <QVariantList>
#include <atomic>

// evaluate() is a pure decision; measure() runs the real engine on a private QRhi.
class FrameGenerationBenchmarkTest : public QObject
{
    Q_OBJECT
private slots:
    void allowsOnlyFactorsTheBudgetAndDisplayAfford()
    {
        QVariantMap measurement{{QStringLiteral("ok"), true}, {QStringLiteral("p95Ms"), 2.5},
                                {QStringLiteral("engineMaxFactor"), 4}};
        // 60 FPS: 16.67 ms interval, 10 ms budget -> 2x (2.5) and 3x (5.0) and 4x (7.5) fit the GPU;
        // the display decides: 120 Hz allows 2x only, 180 Hz adds 3x, 240 Hz adds 4x.
        auto verdict = FrameGenerationBenchmark::evaluate(measurement, 60, 120.0);
        QCOMPARE(verdict.value(QStringLiteral("allowed")).toList(), QVariantList{2});
        QCOMPARE(verdict.value(QStringLiteral("reasons")).toMap().value(QStringLiteral("3")).toString(),
                 QStringLiteral("display"));
        verdict = FrameGenerationBenchmark::evaluate(measurement, 60, 180.0);
        QCOMPARE(verdict.value(QStringLiteral("allowed")).toList(), (QVariantList{2, 3}));
        verdict = FrameGenerationBenchmark::evaluate(measurement, 60, 240.0);
        QCOMPARE(verdict.value(QStringLiteral("allowed")).toList(), (QVariantList{2, 3, 4}));
        // Unknown refresh skips the display gate; the GPU budget still rules.
        measurement.insert(QStringLiteral("p95Ms"), 6.0);
        verdict = FrameGenerationBenchmark::evaluate(measurement, 60, 0.0);
        QCOMPARE(verdict.value(QStringLiteral("allowed")).toList(), QVariantList{2});
        QCOMPARE(verdict.value(QStringLiteral("reasons")).toMap().value(QStringLiteral("3")).toString(),
                 QStringLiteral("gpu"));
        // A 120 FPS source halves the budget: 6 ms per generated frame no longer fits even 2x.
        verdict = FrameGenerationBenchmark::evaluate(measurement, 120, 360.0);
        QVERIFY(verdict.value(QStringLiteral("allowed")).toList().isEmpty());
        // The block-matching engine only synthesises midpoints.
        measurement.insert(QStringLiteral("p95Ms"), 1.0);
        measurement.insert(QStringLiteral("engineMaxFactor"), 2);
        verdict = FrameGenerationBenchmark::evaluate(measurement, 60, 360.0);
        QCOMPARE(verdict.value(QStringLiteral("allowed")).toList(), QVariantList{2});
        QCOMPARE(verdict.value(QStringLiteral("reasons")).toMap().value(QStringLiteral("4")).toString(),
                 QStringLiteral("engine"));
        // Without a measurement nothing is allowed.
        verdict = FrameGenerationBenchmark::evaluate({}, 60, 240.0);
        QVERIFY(verdict.value(QStringLiteral("allowed")).toList().isEmpty());
    }

    void measuresTheEngineOnThisGpu()
    {
        std::atomic_bool cancelled = false;
        double lastProgress = 0.0;
        const auto result = FrameGenerationBenchmark::measure(
            {{QStringLiteral("engine"), QStringLiteral("motion")}, {QStringLiteral("width"), 640},
             {QStringLiteral("height"), 360}, {QStringLiteral("factor"), 2}, {QStringLiteral("seconds"), 1.0}},
            &cancelled, [&lastProgress](double value) { lastProgress = value; });
        if (!result.value(QStringLiteral("ok")).toBool())
            QSKIP(qPrintable(QStringLiteral("GPU benchmark unavailable here: ")
                             + result.value(QStringLiteral("error")).toString()));
        QVERIFY(result.value(QStringLiteral("frames")).toInt() >= 10);
        QVERIFY(result.value(QStringLiteral("p50Ms")).toDouble() > 0.0);
        QVERIFY(result.value(QStringLiteral("p95Ms")).toDouble() >= result.value(QStringLiteral("p50Ms")).toDouble());
        QVERIFY(result.value(QStringLiteral("maxMs")).toDouble() >= result.value(QStringLiteral("p95Ms")).toDouble());
        QVERIFY(!result.value(QStringLiteral("gpu")).toString().isEmpty());
        QCOMPARE(result.value(QStringLiteral("engineMaxFactor")).toInt(), 2);
        QVERIFY(lastProgress > 0.0);
        qInfo() << "capability test:" << result.value(QStringLiteral("gpu")).toString()
                << result.value(QStringLiteral("backend")).toString()
                << "p50" << result.value(QStringLiteral("p50Ms")).toDouble() << "ms"
                << "p95" << result.value(QStringLiteral("p95Ms")).toDouble() << "ms"
                << "gpuTimed" << result.value(QStringLiteral("gpuTimed")).toBool();
    }

    void cancellationStopsTheMeasurement()
    {
        std::atomic_bool cancelled = true;
        const auto result = FrameGenerationBenchmark::measure(
            {{QStringLiteral("width"), 320}, {QStringLiteral("height"), 180}}, &cancelled, {});
        QVERIFY(!result.value(QStringLiteral("ok")).toBool());
    }
};

QTEST_MAIN(FrameGenerationBenchmarkTest)
#include "tst_framegenerationbenchmark.moc"
