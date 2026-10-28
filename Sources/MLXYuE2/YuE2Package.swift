import Foundation
import MLX
import MLXToolKit
import YuE2Core

/// YuE2-3B (M-A-P, CC BY-NC 4.0 weights) on the canonical `music` surface (contract 1.56.0, AB-A-0201): a style caption + lyrics
/// with section tags → a chord-annotated ABC score (the editable plan) → one 48 kHz stereo `.wav` song that sings it. A
/// RESEARCH-ONLY, NAMED-ONLY offer (AB-D-0073 reversed 2026-10-24, ask AB-A-0200, task AB-T-0350): the weights are CC BY-NC 4.0
/// with M-A-P's individual-creator permission, so the host never makes it the capability's default (`MLXServeEngine.setDefault`
/// names ACE-Step for `music`), the operator accepts the terms before first use, and ML[X] Audio Studio — permissive-only
/// (AB-D-0060) — never offers it. The offer's strings (`YuE2Licence`, decided by the operator, AB-D-0161) are the only place the
/// fleet states them.
///
/// Engine-owned lifecycle (C13): the engine constructs from a `YuE2Configuration`, materializes the two declared bundles into its
/// store (or the host names them), pages them in with `load()`, drives `run(_:)`, and reclaims with `unload()`.
///
/// What the surface declares (`MusicControls`) — exactly what V2 gated against the fp32 oracle, nothing more:
///   • the core: caption (the style tags), lyrics (`[verse]` / `[chorus]` / … inline), seed, and `durationSeconds` as a CAP on
///     the semantic tokens (25 per second, at most 9 000 = 360 s): the song ends on its own MUSIC_END when the lyrics run out,
///     and `truncated` says when the cap — or the 4 096-token score budget — cut it instead;
///   • `.planning` with modes `.full` (a chord-annotated ABC score first — the default, gated on the zh golden) and `.off` (no
///     score, two-branch CFG 1.01 — gated on the en golden). `.melody` is upstream's third mode and is NOT declared: no golden
///     covers it (the engine refuses it before admission);
///   • `.score`: a supplied ABC score is rendered instead of planned (the teacher-forced A lane runs exactly that prefix; the T
///     lane round-trips the score's ids). Every `.full` run returns its score on `MusicResponse.score`; an `.off` run has none;
///   • `.guidanceScale`: the semantic stage's CFG (0…20; default 1.0 with a score, 1.01 without — upstream's).
/// Nothing else is honoured, so nothing else is declared and the engine refuses it before admission (C11): no `.vocalLanguage`
/// (the model sings the lyrics' script), no `.anchors` (put tempo and key in the caption — the planner writes them into the score),
/// no `.referenceAudio` / `.sourceAudio` (the VAE encoder is not ported), no `.timestamps`. An instrumental is not gated either:
/// empty lyrics are refused.
///
/// Sampling: both AR phases SAMPLE from `key(seed)` (MLXRandom), so a seed reproduces a song within Swift-MLX, never across
/// bindings (AB-L-0090); the acoustic noise is drawn from the same seed.
@InferenceActor
public final class YuE2Package: ModelPackage {
    public typealias Configuration = YuE2Configuration

    /// Footprints MEASURED as phys_footprint (M5 Max, 2026-10-28, MLX cache 1 GiB, pool-inclusive as the fleet declares —
    /// AB-L-0113 / AB-L-0155): resident = phys after `prepare` through MLXServeEngine (the `validate` lane), activation = the
    /// larger of the validate songs' peak and the `envelope` lane's peak (a 4 096-id score prefix + 9 000 codec frames: the 360 s
    /// input envelope) minus that resident (MEASUREMENTS.md §V3). Declared rounded up:
    ///   bf16 7.60 GB resident, peak 14.39 (song) / 14.58 (envelope) → 8.0 + 7.5;
    ///   int8 (8-bit AR + NAR) 5.08, peak 11.32 / 11.92 → 5.5 + 7.5;
    ///   ar8 (8-bit AR, bf16 NAR) 6.40, peak 12.36 / 13.27 → 7.0 resident hint on the shared `.int8` activation.
    /// The activation is the NAR prefill + solve over ≤ 13 k AR tokens and the VAE tiles, not the weights, so it is the same
    /// ~7 GB on every tier: a 24 GB Mac runs every tier by the engine's 0.7 × RAM rule, a 16 GB Mac none.
    nonisolated static let bf16ResidentBytes: UInt64 = 8_000_000_000
    nonisolated static let bf16PeakActivationBytes: UInt64 = 7_500_000_000
    nonisolated static let int8ResidentBytes: UInt64 = 5_500_000_000
    nonisolated static let int8PeakActivationBytes: UInt64 = 7_500_000_000
    nonisolated static let ar8ResidentBytes: UInt64 = 7_000_000_000

    /// Upstream's semantic budget: 9 000 codec tokens at 25 Hz.
    public nonisolated static let maxDurationSeconds: Double = 360
    public nonisolated static let tokensPerSecond: Double = 25
    public nonisolated static let sampleRate = 48_000
    public nonisolated static let channels = 2

    public nonisolated static var controls: MusicControls {
        MusicControls(features: [.planning, .score, .guidanceScale], planningModes: [.full, .off],
                      maxDurationSeconds: maxDurationSeconds)
    }

    /// The offer fields a media-engine adapter reads (AUDIO-EVALUATION §4.7.5, AB-D-0161) — not expressible on `PackageManifest`.
    public nonisolated static let namedOnly = YuE2Licence.namedOnly
    public nonisolated static let acceptedRequired = YuE2Licence.acceptedRequired
    public nonisolated static let commercialUse = YuE2Licence.commercialUse
    public nonisolated static let weightLicense = SPDXLicense(YuE2Licence.spdx)

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // C7: the weights (m-a-p/YuE2-3B, YuE2-Vae) are CC BY-NC 4.0 — `.rejectedWeight` under `.permissiveOnly`, NEVER
            // allowlisted (a recorded advisory under the default `.advisory` enforcement, a refusal under `.blocking`); both
            // licence texts (HF LICENSE, GitHub MODEL_LICENSE with the individual-creator permission) and the attribution ship
            // with every bundle. C8: port code MIT (upstream code Apache-2.0, Oobleck/SnakeBeta MIT — THIRD_PARTY_NOTICES.md).
            license: LicenseDeclaration(
                weightLicense: weightLicense, portCodeLicense: .mit,
                notices: [
                    // The terms' own restriction and attribution: shown with every result (AUDIO-EVALUATION A-8), accepted before
                    // first use.
                    LicenseNotice(text: YuE2Licence.notice, display: .required, license: weightLicense,
                                  termsURL: YuE2Licence.termsURL),
                    LicenseNotice(text: YuE2Licence.notes, display: .informational),
                    LicenseNotice(text: "A research-only offer: named only (never a capability's default), and the operator accepts "
                                  + "the terms before first use. Commercial use is conditional: allowed for individuals acting in a "
                                  + "personal capacity under M-A-P's MODEL_LICENSE, not for companies or hosted services.",
                                  display: .informational, license: weightLicense, termsURL: YuE2Licence.modelLicenseURL),
                ]),
            provenance: Provenance(sourceRepo: YuE2Configuration.upstreamModelRepo,
                                   revision: YuE2Configuration.upstreamModelRevision, tier: 3),
            requirements: RequirementsManifest(
                footprints: [
                    QuantFootprint(quant: .bf16, residentBytes: bf16ResidentBytes, peakActivationBytes: bf16PeakActivationBytes),
                    // `.int8` is the 8-bit-both tier; `.ar8` shares the case and hints its own resident bytes.
                    QuantFootprint(quant: .int8, residentBytes: int8ResidentBytes, peakActivationBytes: int8PeakActivationBytes),
                ],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: nil
            ),
            specialties: [],
            surfaces: [
                MusicContract.descriptor(
                    name: "yue2-3b",
                    summary: "YuE2-3B (M-A-P) song generator (.wav, 48 kHz stereo, up to 360 s) — NON-COMMERCIAL weights (CC BY-NC 4.0; "
                        + "individuals may sell their songs under M-A-P's individual-creator permission, companies and hosted services may "
                        + "not), offered by name only. A style caption (genre, mood, instruments, voice, tempo) + lyrics with [verse] / "
                        + "[chorus] / [bridge] tags → planning `full`: a chord-annotated ABC score first, returned as `score` to edit and "
                        + "re-render via the `score` parameter; planning `off`: straight to audio. durationSeconds is a cap — the song ends "
                        + "when the lyrics do, and `truncated` says when the cap cut it. guidanceScale: the CFG of the codec-token stage "
                        + "(1.0 with a score, 1.01 without). No language, anchor, reference or edit controls.",
                    controls: Self.controls
                )
            ]
        )
    }

    /// Every V2b tier that passed its pre-registered rule is offered; 4-bit AR failed (and has no case).
    public nonisolated static let offeredTiers: [YuE2Tier] = [.bf16, .ar8, .int8]

    private let configuration: Configuration
    private var pipeline: YuE2Pipeline?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    // MARK: - Lifecycle

    public func load() async throws {
        guard pipeline == nil else { return }
        // Materialization is ENGINE-EXECUTED (contract 1.24): this guard is the offline backstop only.
        let storeRoot = configuration.modelsRootDirectory
        let missing = configuration.missingWeightSources(storeRoot: storeRoot)
        guard missing.isEmpty else {
            throw YuE2PackageError.missingWeights(
                "source not materialized: \(missing.map(\.repo).joined(separator: ", ")) "
                + (storeRoot.map { "(store: \($0.path))" } ?? "(no models root set)"))
        }
        try Task.checkCancellation()
        guard let bundles = configuration.resolvedBundles(storeRoot: storeRoot) else {
            throw YuE2PackageError.missingWeights("unresolved bundle directories (no store root)")
        }
        MLX.Memory.cacheLimit = Int(configuration.cacheLimitBytes)
        // Weight loads ride the CPU stream inside the Core (WeightIO.load) — the Metal-watchdog rule; the tier's quantised
        // parameters are evaluated in `YuE2Quantization.apply`, so the bf16 source Linears are released before the engine reads
        // the footprint.
        do {
            pipeline = try YuE2Pipeline.load(modelDir: bundles.model, vaeDir: bundles.vae, dtype: .bfloat16,
                                             quantization: configuration.tier.quantization)
        } catch let error as YuE2Error {
            throw YuE2PackageError.loadFailed(error.description)
        }
        MLX.Memory.clearCache()
    }

    public func unload() async {
        pipeline = nil
        MLX.Memory.clearCache()   // release the retained MLX pool so eviction frees RSS
    }

    // MARK: - Run

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        // CAN-1: the entry checkpoint is the FIRST act of run(). Mid-run cadence: every score token (plan), every codec token
        // (generate), every ODE step (denoise), every VAE tile (decode) — the Core's `isCancelled` probe; the CancellationError is
        // rethrown UNCHANGED.
        try Task.checkCancellation()
        guard let pipeline else { throw PackageError.notLoaded }
        guard request.capability == .music, let req = request as? MusicRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }
        let plan = try Self.plan(req)
        try Task.checkCancellation()
        // The length cap is a per-request bound on the semantic phase (upstream's `max_tokens`, 9 000 at most).
        var generation = pipeline.config.generation
        generation.semantic.max_tokens = plan.semanticMaxTokens
        pipeline.generation = generation
        let scoreBudget = generation.abc.max_tokens
        let song: SongResult
        do {
            song = try pipeline.generate(plan.request, progress: { p in
                switch p {
                case .plan(let n): RunProgress.report(.plan, step: n, totalSteps: scoreBudget)
                case .semantic(let n): RunProgress.report(.generate, step: n, totalSteps: plan.semanticMaxTokens)
                case .synthesize(let s, let total): RunProgress.report(.denoise, step: s, totalSteps: total)
                case .decode(let t, let total): RunProgress.report(.decode, step: t, totalSteps: total)
                }
            }, isCancelled: { Task.isCancelled })
        } catch YuE2Error.cancelled {
            throw CancellationError()
        } catch YuE2Error.invalidInput(let why) {
            // e.g. a supplied score whose ids leave the text vocabulary — the request's fault, said legibly
            throw PackageError.unsupportedRequestFeature(why)
        } catch let error as YuE2Error {
            throw YuE2PackageError.generationFailed(error.description)
        }
        try Task.checkCancellation()
        RunProgress.report(.postprocess)
        let wav = YuE2AudioIO.encodeWAV16(interleaved: song.audio, channels: song.channels, sampleRate: song.sampleRate)
        // `truncated` is honest (C3): the score hit its 4 096-token budget or the codec tokens hit the cap before MUSIC_END.
        return MusicResponse(audio: Audio(format: .wav, data: wav, sampleRate: song.sampleRate, channels: song.channels),
                             truncated: song.truncated, score: song.plan.abc)
    }

    /// What the request asks for, judged against the declaration (the engine admits only declared features; a direct caller meets
    /// the same rules here) and against the Core's own bounds, mapped onto the Core's request.
    public struct Plan {
        public let request: SongRequest
        /// The semantic phase's token budget: the length cap × 25 Hz, at most 9 000.
        public let semanticMaxTokens: Int
    }

    public nonisolated static func plan(_ req: MusicRequest) throws -> Plan {
        if let refusal = controls.refusal(for: req) { throw PackageError.unsupportedRequestFeature(refusal) }
        let caption = req.caption.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !caption.isEmpty else {
            throw PackageError.unsupportedRequestFeature("empty caption — the style tags condition both the score and the song")
        }
        let lyrics = req.lyrics.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !lyrics.isEmpty else {
            throw PackageError.unsupportedRequestFeature("empty lyrics — this surface sings lyrics; no instrumental mode is gated")
        }
        let duration = req.durationSeconds ?? maxDurationSeconds
        guard duration > 0 else {
            throw PackageError.unsupportedRequestFeature(
                String(format: "durationSeconds(%g) — the cap must be positive (at most %g s)", duration, maxDurationSeconds))
        }
        var score: String? = nil
        if let s = req.score {
            guard !s.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw PackageError.unsupportedRequestFeature("score — an empty ABC score; omit it to let the model plan one")
            }
            score = s
        }
        let mode: PlanningMode
        switch req.planning {
        case nil, .full?: mode = .full
        case .off?: mode = .off
        case .melody?: throw PackageError.unsupportedRequestFeature("planning(.melody) — not a declared planning mode (full / off)")
        @unknown default: throw PackageError.unsupportedRequestFeature("planning — an unknown mode")
        }
        if score != nil, mode == .off {
            throw PackageError.unsupportedRequestFeature("score with planning(.off) — a supplied score is rendered in planning `full`")
        }
        let cfg = req.guidanceScale.map(Float.init)
        if let c = cfg, !(c.isFinite && c >= 0 && c <= 20) {
            throw PackageError.unsupportedRequestFeature(
                "guidanceScale(\(req.guidanceScale!)) — the codec-token CFG takes 0…20 (1.0 with a score, 1.01 without)")
        }
        let tokens = Int(min(9_000, (duration * tokensPerSecond).rounded(.up)))
        do {
            let r = try SongRequest(style: caption, lyrics: lyrics, mode: mode,
                                    seed: req.seed ?? UInt64.random(in: 0 ... UInt64(UInt32.max)), abc: score, cfgScale: cfg)
            return Plan(request: r, semanticMaxTokens: max(1, tokens))
        } catch let error as YuE2Error {
            throw PackageError.unsupportedRequestFeature(error.description)
        }
    }
}

extension YuE2Package {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration { .of(YuE2Package.self) }
}

/// Wrapper-level errors (weight resolution, a Core failure). Runtime request errors use `PackageError`.
public enum YuE2PackageError: Error, CustomStringConvertible {
    case missingWeights(String)
    case loadFailed(String)
    case generationFailed(String)
    public var description: String {
        switch self {
        case .missingWeights(let s): return "YuE2-3B: \(s)"
        case .loadFailed(let s): return "YuE2-3B: load failed — \(s)"
        case .generationFailed(let s): return "YuE2-3B: \(s)"
        }
    }
}
