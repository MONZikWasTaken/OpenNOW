# FG-Legacy: neural 2× frame generation for the native renderer

FG-Legacy is the Framegen v7-small (v7s) neural frame interpolator ported to the
Qt/QRhi native renderer. It replaces the block-matching midpoint synthesizer in
`StreamFrameInterpolator` behind the same contract. The name is deliberate: this is
the proven, shipped model from the Framegen project, and it is planned to be
replaced later by the FlyHigh project (a new model and runtime built for the same
slot). Nothing outside the interpolator contract changes.

Framegen was already part of OpenNOW once: PR #718/#763 added it to the Electron
client in August. The Qt rebuild removed the Electron tree and, with it, the
integration and its third-party notices. FG-Legacy restores the capability on the
native path and restores the attribution (`third_party/framegen/`,
`THIRD_PARTY_NOTICES`).

## Why replace the current synthesizer

The current frame generation (`framegen_motion.frag` + `framegen_synthesize.frag`)
estimates bidirectional block-matching motion on a pyramid whose base is at most
320×180 and writes the **current frame** into every midpoint pixel whose match is
not reliable. Reliable means: a textured match (second-best cost clearly worse than
best), small residual, colour error below 0.12 and forward/backward agreement
within 1.5 texels; flat regions interpolate only when they move less than a quarter
texel. On compressed game video most pixels fail at least one of these, so the
midpoint is mostly a duplicate of the next frame. That is the live DX11 report in
PR #813, "the counter reaches 120 but visible motion remains at 60", and it is
content-dependent, not backend-dependent; the synthetic translation fixtures pass
because every pixel in them is textured and the motion is tiny. Where the match is
accepted the midpoint is a plain average of two warps, which ghosts at occlusions,
and flow is bilinearly upsampled from 8–12 px blobs, which halos moving edges.

A learned interpolator produces a true midpoint everywhere: dense flow for both
directions, a learned occlusion mask instead of a reliability gate, and a residual
network for the regions where warping cannot be right. It needs no per-content
thresholds.

## The model

v7s is a t-factored one-block IFNet-style network, 736 456 parameters, 2.95 MB of
f32 weights. Weights ship unchanged from the Framegen release (`rt_v7s.bin` +
`rt_v7s.json`, 47 tensors); a fresh export from the training checkpoint is
byte-identical. Channel order inside activation buffers is BGR planes.

```
x    = cat(prev, cur)                                  [6, Hm, Wm]   sampled at model grid
xq   = x at 1/4                                        [6, Hm/4, Wm/4]   (prep pass)
f0   = prelu(conv3x3 s2 (6→48))                        [48, Hm/8, Wm/8]
feat0= prelu(conv3x3 s2 (48→96))                       [96, Hm/16, Wm/16]
h    = 6 × (prelu(conv3x3 s1 (96→96)))  (trunk, t-free)
hm   = h · (1 + s_t) + b_t                             FiLM, (s_t, b_t) = MLP(t) ∈ R^192
hh   = 2 × (prelu(conv3x3 s1 (96→96)))  (head)
tmp  = deconv4x4 s2 (96→5)(hh + feat0)                 [5, Hm/8, Wm/8]
flow = bilinear×8(tmp[0:4]) · 8      mask = sigmoid(bilinear×8(tmp[4]))
w0   = warp(prev, flow[0:2])   w1 = warp(cur, flow[2:4])     (bilinear, clamp-to-edge)
mid  = w0 · mask + w1 · (1 − mask)
rin  = cat(w0, w1, mask, flow · 0.25/20) at 1/4         [11, Hm/4, Wm/4]
res  = bilinear×4( 2·sigmoid(refine(rin)) − 1 )         refine: 11→16→16→16→3, PReLU
out  = clamp(mid + res, 0, 1)
```

The trunk runs once per frame pair; only the FiLM parameters and the head depend on
t. For 2× only t = 0.5 is used, so the whole graph runs once per generated frame.
Model dimensions (Wm, Hm) must be multiples of 16.

Reference cost: 3.0 GMAC at a 1280×720 model grid, 1.3 GMAC at 854×480. The WebGPU
runtime measures 2.6 ms at 480p and 3.8 ms at 720p on an RTX 4060 Ti with f16
storage. An f32 QRhi port lands in the same order; the budget for a 120 Hz output
is 8.3 ms per generated frame.

## QRhi mapping

Everything runs on the scene-graph render thread inside `prepareFrame`, through the
QRhi command buffer, exactly like the current passes. No second runtime, no CPU
readback, no extra presenter, and the pacer, stats and teardown paths are reused.

Backends: D3D11 (cs_5_0), Vulkan (SPIR-V) and Metal, which are the three OpenNOW
forces. `QRhi::Compute` is checked at `initialize()`; when it is unavailable (Null,
OpenGL ES 2) the interpolator reports failure the same way the current one does and
frame generation shows `unavailable`.

Passes per generated frame, in order:

| # | pass | kind | resources |
|---|---|---|---|
| 1 | prep | compute 8×8 | samples `history[prev]`, `history[cur]` at model-grid texel centres (hardware resize), writes `xq` (f32 SSBO), accumulates the scene-cut statistic |
| 2 | conv0a, conv0b | compute | SSBO → SSBO, stride 2, weights in immutable SSBOs |
| 3 | trunk ×6, FiLM, head ×2 | compute | SSBO → SSBO; FiLM parameters are a 192-float uniform computed on the CPU from the MLP (two tiny matrices) |
| 4 | residual + deconv | compute | writes flow (RGBA16F) and mask (R16F) at Hm/8 via `imageLoadStore`; falls back to an SSBO when `UsedWithLoadStore` is unsupported for the format |
| 5 | refine prep | compute | samples flow/mask with the bilinear sampler, warps both history textures, writes `rin` at Hm/4 |
| 6 | refine ×4 | compute | SSBO → residual (RGBA16F at Hm/4, or SSBO) |
| 7 | composite | fragment | into the existing `midpoint` render target at source resolution: sample flow/mask, warp both frames with the clamp-to-edge sampler, blend, add residual, clamp; scene-cut passthrough |

The composite stays a fragment pass so the output keeps the current `Surface`
plumbing and formats (RGBA8, RGB10A2) without storage-image constraints. Warps use
the hardware bilinear sampler with clamp-to-edge addressing, which is the reference
warp (border padding, pixel-space bilinear). Flow is produced on the model grid and
scaled by `source / model` when the stream is larger than the model grid.

Storage buffers are f32. Activations at a 720p model grid total about 8 MB; the
weights are 2.95 MB. Thread groups are 8×8 for image-shaped passes and 256 for
channel-shaped ones, within every backend's limits.

Compute shaders are GLSL 440 `.comp` files baked by `qt_add_shaders` with
`GLSL "430" HLSL 50 MSL 12` (compute needs those versions; the vertex/fragment
default set stays as is).

### Model grid

The model grid is the stream size capped at 1280×720, rounded to multiples of 16,
aspect preserved; a 1080p stream runs the network at 1280×720 and composites at
1080p. The cap is the quality/cost knob: 854×480 for weak GPUs, 1920×1088 for
strong ones. v1 exposes it through `OPENNOW_FG_LEGACY_MODEL_HEIGHT` (480, 720,
1080); a settings entry can follow once the maintainers choose a UI.

### Scene cuts

The network warps garbage across a cut, so the prep pass accumulates the mean
absolute luma difference between the two quarter-resolution inputs into an atomic
counter; the composite pass reads it and passes the current frame through when the
mean exceeds 0.16, matching the threshold of the current cut detector. No CPU
readback is involved.

### Engine selection

For review and A/B the current block-matching engine stays selectable with
`OPENNOW_FRAME_GENERATION_ENGINE=motion`; the default is `fg-legacy` whenever
compute is available. The setting key `frameGeneration` (off / 2×) is unchanged.

## Pacing, latency, HDR

The pacer (`StreamFramePacer`) is untouched by this change; its gates are addressed
in a separate, smaller pull request (display-refresh fallback, jitter-tolerant
overload, 60 Hz path). Latency is the same as today: a midpoint needs both frames,
so generated output trails the source by one source interval plus the compute time.
Frame generation remains SDR-only; the model was trained on SDR content and the HDR
exclusion in `NativeStreamRenderCallback` stays.

## Verification

1. **Kernel parity.** Each compute kernel is checked against PyTorch on the same
   buffers (conv s1/s2 with PReLU and residual, deconv, FiLM, refine), tolerance
   1e-4 relative in f32.
2. **Graph parity.** The full FG-Legacy midpoint for two fixture pairs is compared
   with the PyTorch v7s output: PSNR ≥ 40 dB after the hardware sampler's ±1 LSB,
   flow L1 ≤ 0.05 px on the model grid.
3. **Behaviour.** `tst_frameinterpolator` keeps its contract: the midpoint beats
   source repetition and a crossfade on the synthetic translations, scene cuts pass
   the current frame through, resources survive device loss and size changes. A new
   fixture adds a flat moving region and an occluding object, the two cases the
   block-matching engine cannot interpolate.
4. **Live.** RTX 4060 Ti, D3D11, 239 Hz display, 1080p60 stream: output FPS,
   visible motion at 120 (not 60), ms per generated frame, GPU load, and the same
   measurements with the current engine for comparison.

## Licensing and attribution

The Framegen runtime code is MIT. The v7s model weights are licensed for
non-commercial research and personal use only (Framegen `WEIGHTS_LICENSE.md`), the
same terms OpenNOW shipped under in the Electron client. `third_party/framegen/`
carries both notices, `THIRD_PARTY_NOTICES` gets a Framegen section, and the
settings hint keeps the weights notice.

Upstream: https://github.com/MONZikWasTaken/Framegen

## Levels, target FPS, display refresh and the capability test

Frame generation is no longer a single 2× switch. The settings model has four pieces,
all owned by `SettingsState` (QML) with the Rust core validating every value:

| key | values | meaning |
|---|---|---|
| `frameGeneration` | `off`, `2x`, `3x`, `4x` | displayed frames per source frame (the level) |
| `frameGenerationTargetFps` | 0 or 30…1000 | 0 follows the level; otherwise the requested output rate, and the level is derived as `round(target / source fps)` clamped to 2…4 |
| `frameGenerationDisplayHz` | 0 or 30…1000 | 0 uses the refresh rate of the display OpenNOW is on; otherwise a manual override for displays that report nothing or the wrong value |
| `frameGenerationProfiles` | object, ≤ 64 entries | capability test results keyed by `engine|gpu|WxH|fps` (`fps` is `auto` when the stream rate is Auto) |

The source frame rate generation works against is the selected stream FPS, or for Auto the
highest rate the membership allows at the selected resolution. The pacer takes the factor
and the effective refresh rate; the display gate scales with the factor (`refresh ≥
(N − 0.05) × source`), and the former hard 120 FPS ceiling is gone: a 120 FPS stream on a
360 Hz display may run 2×. Only a 4 ms source interval floor remains. The engine reports how
many frames it can synthesise per pair (`generatedCount()`): the block-matching engine
produces the midpoint only, so 3× and 4× need FG-Legacy, and the schedule follows the
effective factor.

### Capability test

`FrameGenerationBenchmark` (context property) runs the engine offscreen on a private QRhi of
the same backend, at the selected stream resolution, over a ring of procedural frames with
texture, a flat band and an occluding block, for about three seconds. Per generated frame it
records GPU time from `QRhi::Timestamps` (`lastCompletedGpuTime`), falling back to wall time
around the blocking offscreen frame, and reports p50/p95/max plus the GPU name
(`QRhiDriverInfo`). The decision is pure:

- GPU: `p95 × (N − 1) ≤ 0.6 × (1000 / fps)` ms, leaving 40 % of each source interval for
  decode, upscaling and presentation.
- Display: `refresh ≥ fps × N × 0.975` (skipped when the refresh is unknown).
- Engine: `N ≤ engineMaxFactor`.

The result is stored under the preset key, so changing resolution, frame rate or GPU shows
the levels as locked again until that combination is tested. Levels and target rates the
profile does not allow are disabled in the UI with a hint; a selection that the test rules
out is reset to Off. The runtime still honours whatever is stored (the acceptance contract
`applySetting("frameGeneration", "2x")` → surfaces enable is unchanged); the gate is in the
settings UI, which is the only normal path to enable it. The first-run Boost step keeps its
simple Off / 2× choice.

### UI

Desktop (Streaming page) and console (Video settings) show the same four controls: level
segments, a **Run test** button, **Target FPS** (Level, or the rates the display can show
at 2…4× the stream rate), **Display refresh** (Auto with the detected value, or an
override), and a GitHub icon that opens the Framegen repository. The stats overlay badge
reads `FRAME GEN N×`.

## Out of scope for v1

Multipliers above 2×, extrapolation, HDR sources, mobile targets and the f16
storage path. The FlyHigh project will later replace FG-Legacy in the same slot with
a new model and a runtime that executes an exported operator graph.
