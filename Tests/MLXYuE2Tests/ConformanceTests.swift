// ConformanceTests.swift — the offline conformance gate for MLXYuE2: manifest (C1 / C7 / C8 / C-memory / C11 / provenance, the
// CC-BY-NC-4.0 weights never admitted by the permissive policies, the offer's notices), the MAT gate per tier (auto-
// materialization declarations, two roles), the CAN gate (pre-cancelled run propagation, checkpoint cadence), the unloaded refusal,
// and the request plane (the core fields, the declared planning / score / cfg, every undeclared feature refused). Nothing here
// evals a kernel: the live run is `yue2-gates validate` (GPU, through MLXServeEngine).

import Foundation
import MLX
import MLXServeConformance
import MLXServeCore
import MLXToolKit
import XCTest

@testable import MLXYuE2
@testable import YuE2Core

final class ManifestConformanceTests: XCTestCase {

    /// C7/C8 — CC-BY-NC-4.0 weights (an SPDX id), MIT port code; NOT admitted by `.permissiveOnly` nor by the eval policy (never
    /// allowlisted, so the Studio can never load it), admitted only under `.any`; the offer's notice is the operator's text.
    func testLicenseIsNonCommercialAndNeverAllowlisted() {
        let license = YuE2Package.manifest.license
        XCTAssertEqual(license.weightLicense, SPDXLicense("CC-BY-NC-4.0"))
        XCTAssertNil(license.additionalWeightLicenses)
        XCTAssertEqual(license.portCodeLicense, .mit)
        XCTAssertEqual(LicensePolicy.permissiveOnly.evaluate(license), .rejectedWeight("CC-BY-NC-4.0"))
        XCTAssertEqual(LicensePolicy.permissiveOrAcknowledged.evaluate(license), .rejectedWeight("CC-BY-NC-4.0"))
        XCTAssertTrue(LicensePolicy.any.evaluate(license).isAdmitted)
        XCTAssertFalse(SPDXLicense.permissiveAllowlist.contains(license.weightLicense))
        XCTAssertFalse(SPDXLicense.evalAcknowledgedAllowlist.contains(license.weightLicense))
        XCTAssertTrue(license.missingRequiredDisplayNotices.isEmpty)
        // the notice is shown with every result and carries the individual-creator condition (AB-D-0161) and the attribution
        XCTAssertEqual(license.requiredNotices.count, 1)
        let notice = license.requiredNotices[0]
        XCTAssertEqual(notice.text, YuE2Licence.notice)
        XCTAssertTrue(notice.text.contains("YuE2-3B by M-A-P (huggingface.co/m-a-p/YuE2-3B), CC BY-NC 4.0"))
        XCTAssertTrue(notice.text.contains("individual-creator permission"))
        XCTAssertTrue(notice.text.contains("Companies may not use the model commercially"))
        XCTAssertEqual(notice.termsURL, "https://huggingface.co/m-a-p/YuE2-3B/blob/main/LICENSE")
        XCTAssertEqual(license.notices?.count, 3)
        let informational = license.notices?.filter { $0.display == .informational }.map(\.text) ?? []
        XCTAssertTrue(informational.contains(YuE2Licence.notes))
        XCTAssertTrue(informational.joined().contains("named only"))
        // the offer fields an MES adapter reads
        XCTAssertTrue(YuE2Package.namedOnly)
        XCTAssertTrue(YuE2Package.acceptedRequired)
        XCTAssertEqual(YuE2Package.commercialUse, "conditional")
    }

    /// Under `.blocking` the default policy refuses to register it; under the default `.advisory` it registers with ONE weight
    /// advisory naming the licence.
    func testEngineGateClassifiesTheWeights() async throws {
        let blocking = MLXServeEngine(licenseEnforcement: .blocking)
        do {
            _ = try await blocking.register(YuE2Package.registration, configuration: YuE2Configuration(), id: PackageID("yue2-3b"))
            XCTFail("registered under .blocking .permissiveOnly")
        } catch {}
        let advisory = MLXServeEngine()
        _ = try await advisory.register(YuE2Package.registration, configuration: YuE2Configuration(), id: PackageID("yue2-3b"))
        let advisories = await advisory.licenseAdvisories
        XCTAssertEqual(advisories.count, 1)
        XCTAssertEqual(advisories.first?.layer, .weight)
        XCTAssertEqual(advisories.first?.license, "CC-BY-NC-4.0")
    }

    /// C-memory — split footprints for the bf16 and 8-bit tiers, at least what `validate` / `envelope` measured; `.ar8` hints its
    /// own resident bytes between the two.
    func testFootprintsAreTheMeasuredTiers() {
        let footprints = YuE2Package.manifest.requirements.footprints
        XCTAssertEqual(footprints.map(\.quant), [.bf16, .int8])
        // ≥ the phys_footprint readings of the V3 validate / envelope lanes (MEASUREMENTS §V3)
        XCTAssertGreaterThanOrEqual(footprints[0].residentBytes, 7_600_000_000)                                  // bf16 after prepare
        XCTAssertGreaterThanOrEqual(footprints[0].residentBytes + footprints[0].peakActivationBytes, 14_580_000_000)   // bf16 envelope peak
        XCTAssertGreaterThanOrEqual(footprints[1].residentBytes, 5_080_000_000)                                  // int8 after prepare
        XCTAssertGreaterThanOrEqual(footprints[1].residentBytes + footprints[1].peakActivationBytes, 11_920_000_000)   // int8 envelope peak
        XCTAssertGreaterThanOrEqual(YuE2Package.ar8ResidentBytes, 6_400_000_000)                                 // ar8 after prepare
        XCTAssertGreaterThanOrEqual(YuE2Package.ar8ResidentBytes + footprints[1].peakActivationBytes, 13_270_000_000)  // ar8 envelope peak
        XCTAssertLessThan(footprints[1].residentBytes, footprints[0].residentBytes)
        XCTAssertNil(YuE2Configuration().residentBytesHint)
        XCTAssertNil(YuE2Configuration(tier: .int8).residentBytesHint)
        XCTAssertEqual(YuE2Configuration(tier: .ar8).residentBytesHint, YuE2Package.ar8ResidentBytes)
        XCTAssertGreaterThan(YuE2Package.ar8ResidentBytes, footprints[1].residentBytes)
        XCTAssertLessThan(YuE2Package.ar8ResidentBytes, footprints[0].residentBytes)
        XCTAssertTrue(CancellationConformance.longRunImplied(by: YuE2Package.manifest))
        XCTAssertEqual(YuE2Package.offeredTiers, [.bf16, .ar8, .int8])
        XCTAssertEqual(YuE2Configuration(tier: .ar8).quant, .int8)
        XCTAssertEqual(YuE2Tier.ar8.quantization.arBits, 8)
        XCTAssertNil(YuE2Tier.ar8.quantization.narBits)
        XCTAssertEqual(YuE2Tier.int8.quantization.narBits, 8)
        XCTAssertNil(YuE2Tier.bf16.quantization.arBits)
    }

    /// C1 / C11 — the package serves `music` and nothing else; the surface declares exactly what V2 gated: planning (full / off),
    /// a supplied score, the CFG scale, a 360 s cap; nothing else is advertised.
    func testCapabilityIsMusicWithAnHonestDeclaration() {
        XCTAssertEqual(Set(YuE2Package.manifest.capabilities), [.music])
        let surface = YuE2Package.manifest.surfaces.first { $0.capability == .music }
        XCTAssertEqual(surface?.name, "yue2-3b")
        let controls = surface?.musicControls
        XCTAssertEqual(controls?.features, [.planning, .score, .guidanceScale])
        XCTAssertEqual(controls?.planningModes, [.full, .off])
        XCTAssertEqual(controls?.editModes, [])
        XCTAssertEqual(controls?.vocalLanguages, [])
        XCTAssertEqual(controls?.maxDurationSeconds, 360)
        XCTAssertEqual(controls?.declarationProblems, [])
        let names = Set(surface?.parameters.map(\.name) ?? [])
        XCTAssertEqual(names, ["caption", "lyrics", "durationSeconds", "seed", "guidanceScale", "planning", "score"])
        XCTAssertTrue(surface?.summary.contains("NON-COMMERCIAL") ?? false)
        XCTAssertTrue(surface?.summary.contains("by name only") ?? false)
    }

    /// C0 — the manifest targets a contract that has the music capability (1.56.0 or later).
    func testContractVersionHasTheMusicCapability() {
        let v = YuE2Package.manifest.contractVersion
        XCTAssertTrue(v.major > 1 || (v.major == 1 && v.minor >= 56), "\(v)")
    }

    func testProvenancePinsTheUpstreamCheckpoint() {
        XCTAssertEqual(YuE2Package.manifest.provenance.sourceRepo, "m-a-p/YuE2-3B")
        XCTAssertEqual(YuE2Package.manifest.provenance.revision, YuE2Configuration.upstreamModelRevision)
        XCTAssertEqual(YuE2Package.manifest.provenance.revision.count, 40)
        XCTAssertEqual(YuE2Package.manifest.provenance.tier, 3)
    }
}

final class MaterializationConformanceTests: XCTestCase {

    /// MAT-1..5: a fresh (dir-less) configuration reports both sources missing; explicit bundles satisfy — per offered tier.
    func testMaterializationGatePerTier() throws {
        for tier in YuE2Package.offeredTiers {
            let (satisfied, root) = try satisfiedConfiguration(tier: tier)
            defer { try? FileManager.default.removeItem(at: root) }
            let report = MaterializationConformance.check(
                freshConfiguration: YuE2Configuration(tier: tier, modelsRootDirectory: emptyStoreRoot()),
                satisfiedConfiguration: satisfied)
            XCTAssertTrue(report.passed, "\(tier): \(report.summary)")
        }
    }

    /// Two roles: the backbone bundle and the VAE decoder bundle (the V4-published repos), the same on every tier (tiers quantise
    /// at load).
    func testWeightSources() {
        for tier in YuE2Tier.allCases {
            let c = YuE2Configuration(tier: tier)
            XCTAssertEqual(c.weightSources.map(\.role), ["model", "vae"])
            XCTAssertEqual(c.weightSources.map(\.repo), ["mlx-community/YuE2-3B-bf16", "mlx-community/YuE2-Vae-fp32"])
            XCTAssertEqual(c.weightSources[0].matching, YuE2Configuration.modelFiles)
            XCTAssertEqual(c.weightSources[1].matching, YuE2Configuration.vaeFiles)
        }
        XCTAssertEqual(YuE2Configuration(revision: "abc", vaeRevision: "def").weightSources.map(\.revision), ["abc", "def"])
        // nil = the V4-published commits (hash-verified against the gated bundles)
        XCTAssertEqual(YuE2Configuration().weightSources.map(\.revision),
                       ["7024750ff947a1c57191a34908e14a8f55548516", "486147eeb1e8397a87a7545ab51d9df95db60a4c"])
        XCTAssertEqual(YuE2Configuration().cacheLimitBytes, 1 << 30)
    }

    /// The declared globs are the files `YuE2Pipeline.load` opens, in the converter's layout.
    func testDeclaredFilesMatchTheBundleLayout() {
        let model = ["config.json", "model.safetensors", "qwen.tiktoken", "LICENSE", "MODEL_LICENSE", "NOTICE"]
        let vae = ["config.json", "decoder.safetensors", "LICENSE", "MODEL_LICENSE", "NOTICE"]
        for glob in YuE2Configuration.modelFiles {
            XCTAssertTrue(model.contains { WeightSourceProbe.matches(path: $0, glob: glob) }, "model glob \(glob)")
        }
        for glob in YuE2Configuration.vaeFiles {
            XCTAssertTrue(vae.contains { WeightSourceProbe.matches(path: $0, glob: glob) }, "vae glob \(glob)")
        }
    }

    /// An explicit pair reads as missing per role until its probe file exists; resolution needs both.
    func testExplicitBundleProbe() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("yue2-partial-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let m = root.appendingPathComponent("model"), v = root.appendingPathComponent("vae")
        try FileManager.default.createDirectory(at: m, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: v, withIntermediateDirectories: true)
        let c = YuE2Configuration(modelDirectory: m, vaeDirectory: v)
        XCTAssertEqual(c.missingWeightSources(storeRoot: nil).map(\.role), ["model", "vae"])
        try Data().write(to: m.appendingPathComponent(YuE2Configuration.modelProbe))
        XCTAssertEqual(c.missingWeightSources(storeRoot: nil).map(\.role), ["vae"])
        try Data().write(to: v.appendingPathComponent(YuE2Configuration.vaeProbe))
        XCTAssertTrue(c.missingWeightSources(storeRoot: nil).isEmpty)
        XCTAssertEqual(c.resolvedBundles(storeRoot: nil)?.model, m)
        XCTAssertEqual(c.resolvedBundles(storeRoot: nil)?.vae, v)
        XCTAssertEqual(c.prewarmPaths.count, 2)
        XCTAssertNil(YuE2Configuration().resolvedBundles(storeRoot: nil))
        XCTAssertEqual(YuE2Configuration(modelDirectory: m).missingWeightSources(storeRoot: nil).map(\.role), ["vae"])
    }

    func testCodableExcludesEnvironmentURLs() throws {
        let c = YuE2Configuration(tier: .ar8, revision: "abc", vaeRevision: "def", modelDirectory: URL(fileURLWithPath: "/tmp/x"),
                                  vaeDirectory: URL(fileURLWithPath: "/tmp/v"), modelsRootDirectory: URL(fileURLWithPath: "/tmp/y"),
                                  cacheLimitBytes: 2 << 30)
        let decoded = try JSONDecoder().decode(YuE2Configuration.self, from: JSONEncoder().encode(c))
        XCTAssertNil(decoded.modelDirectory)
        XCTAssertNil(decoded.vaeDirectory)
        XCTAssertNil(decoded.modelsRootDirectory)
        XCTAssertEqual(decoded.tier, .ar8)
        XCTAssertEqual(decoded.revision, "abc")
        XCTAssertEqual(decoded.vaeRevision, "def")
        XCTAssertEqual(decoded.cacheLimitBytes, 2 << 30)
        let old = try JSONDecoder().decode(YuE2Configuration.self, from: Data("{}".utf8))
        XCTAssertEqual(old.tier, .bf16)
    }

    private func emptyStoreRoot() -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("yue2-empty-store-\(UUID().uuidString)")
    }

    private func satisfiedConfiguration(tier: YuE2Tier) throws -> (YuE2Configuration, URL) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("yue2-explicit-\(UUID().uuidString)")
        let m = root.appendingPathComponent("model"), v = root.appendingPathComponent("vae")
        try FileManager.default.createDirectory(at: m, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: v, withIntermediateDirectories: true)
        try Data().write(to: m.appendingPathComponent(YuE2Configuration.modelProbe))
        try Data().write(to: v.appendingPathComponent(YuE2Configuration.vaeProbe))
        return (YuE2Configuration(tier: tier, modelDirectory: m, vaeDirectory: v), root)
    }
}

final class CancellationConformanceTests: XCTestCase {

    private func probe() -> MusicRequest {
        MusicRequest(caption: "cancellation probe", lyrics: "[verse]\nla la la", durationSeconds: 30, seed: 1)
    }

    /// CAN-1/CAN-2 — a pre-cancelled `run()` surfaces `CancellationError` unchanged, even unloaded.
    func testPreCancelledRunPropagatesCancellation() async {
        let package = YuE2Package(configuration: YuE2Configuration())
        let report = await CancellationConformance.checkRun(package: package, request: probe())
        XCTAssertTrue(report.passed, report.summary)
    }

    /// CAN-3 — checkpoints at every score token (plan), every codec token (generate), every ODE step (denoise) and every VAE tile
    /// (decode), each on the same seam as the RunProgress report.
    func testCheckpointCadence() {
        let report = CancellationConformance.checkCadence(
            manifest: YuE2Package.manifest,
            posture: .cadence([.init(phase: .plan, unit: .token, reportsRunProgress: true),
                               .init(phase: .generate, unit: .token, reportsRunProgress: true),
                               .init(phase: .denoise, unit: .step, reportsRunProgress: true),
                               .init(phase: .decode, unit: .chunk, reportsRunProgress: true)]))
        XCTAssertTrue(report.passed, report.summary)
    }
}

final class RequestPlaneTests: XCTestCase {

    private func request(duration: Double? = 120, lyrics: String = "[verse]\nla la la", language: String? = nil,
                         anchors: MusicAnchors? = nil, planning: MusicPlanningMode? = nil, score: String? = nil,
                         edit: MusicEdit? = nil, reference: Audio? = nil, guidance: Double? = nil) -> MusicRequest {
        MusicRequest(caption: "warm piano pop, female vocal", lyrics: lyrics, durationSeconds: duration, seed: 7, guidanceScale: guidance,
                     planning: planning, score: score, referenceAudio: reference, edit: edit, anchors: anchors, vocalLanguage: language)
    }

    func testUnloadedPackageRejectsLegibly() async {
        let package = YuE2Package(configuration: YuE2Configuration())
        do {
            _ = try await package.run(request())
            XCTFail("expected notLoaded")
        } catch let error as PackageError {
            guard case .notLoaded = error else { return XCTFail("expected notLoaded, got \(error)") }
        } catch {
            XCTFail("expected PackageError, got \(error)")
        }
    }

    /// The core maps 1:1 onto the Core's request: planning defaults to `full`, the cap becomes the semantic budget (25 Hz, ≤ 9 000),
    /// a nil length is the full 360 s budget, a nil seed is drawn, the cfg is the mode's default unless given.
    func testCoreMapsOntoThePipeline() throws {
        let p = try YuE2Package.plan(request())
        XCTAssertEqual(p.request.style, "warm piano pop, female vocal")
        XCTAssertEqual(p.request.lyrics, "[verse]\nla la la")
        XCTAssertEqual(p.request.mode, .full)
        XCTAssertEqual(p.request.seed, 7)
        XCTAssertNil(p.request.abc)
        XCTAssertEqual(p.request.guidance, 1.0)
        XCTAssertEqual(p.semanticMaxTokens, 3000)
        XCTAssertEqual(try YuE2Package.plan(request(duration: nil)).semanticMaxTokens, 9000)
        XCTAssertEqual(try YuE2Package.plan(request(duration: 360)).semanticMaxTokens, 9000)
        XCTAssertEqual(try YuE2Package.plan(request(duration: 0.01)).semanticMaxTokens, 1)
        XCTAssertEqual(try YuE2Package.plan(request(duration: 84.3)).semanticMaxTokens, 2108)
        let off = try YuE2Package.plan(request(planning: .off))
        XCTAssertEqual(off.request.mode, .off)
        XCTAssertEqual(off.request.guidance, 1.01)
        XCTAssertEqual(try YuE2Package.plan(request(guidance: 1.2)).request.guidance, 1.2)
        let scored = try YuE2Package.plan(request(planning: .full, score: "X:1\nK:C\nCDEF|"))
        XCTAssertEqual(scored.request.abc, "X:1\nK:C\nCDEF|")
        XCTAssertEqual(try YuE2Package.plan(request(score: "X:1\nK:C\nCDEF|")).request.mode, .full)
        XCTAssertNotEqual(try YuE2Package.plan(MusicRequest(caption: "c", lyrics: "l")).request.seed, 0)
        XCTAssertNoThrow(try YuE2Package.plan(request(anchors: MusicAnchors())))   // an empty anchor asks for nothing
    }

    /// Every undeclared feature and out-of-range value is refused legibly (the engine refuses the undeclared ones before admission).
    func testUndeclaredFeaturesAndBoundsAreRefused() {
        let bad: [(String, MusicRequest)] = [
            ("melody", request(planning: .melody)),
            ("edit", request(edit: MusicEdit(source: Audio(format: .wav, data: Data([1])), operation: .cover(strength: 0.5)))),
            ("reference", request(reference: Audio(format: .wav, data: Data([1])))),
            ("anchors", request(anchors: MusicAnchors(bpm: 112))),
            ("language", request(language: "en")),
            ("too long", request(duration: 400)),
            ("zero length", request(duration: 0)),
            ("empty caption", MusicRequest(caption: "  ", lyrics: "la")),
            ("empty lyrics", request(lyrics: "  \n")),
            ("empty score", request(score: "  ")),
            ("score + off", request(planning: .off, score: "X:1\nK:C\nCDEF|")),
            ("cfg 21", request(guidance: 21)),
            ("cfg -1", request(guidance: -1)),
            ("cfg nan", request(guidance: .nan)),
        ]
        for (label, r) in bad {
            do { _ = try YuE2Package.plan(r); XCTFail("admitted: \(label)") }
            catch let e as PackageError { guard case .unsupportedRequestFeature = e else { return XCTFail("\(label): \(e)") } }
            catch { XCTFail("\(label): \(error)") }
        }
        // the engine-side refusal is the declaration's own: melody / edit / reference / anchors / language / length
        for r in [request(planning: .melody), request(language: "en"), request(duration: 400), request(anchors: MusicAnchors(key: "C major"))] {
            XCTAssertNotNil(YuE2Package.controls.refusal(for: r))
        }
        XCTAssertNil(YuE2Package.controls.refusal(for: request(planning: .off, guidance: 3)))
        XCTAssertNil(YuE2Package.controls.refusal(for: request(score: "X:1")))
    }

    func testWAVEncoding() {
        let data = YuE2AudioIO.encodeWAV16(interleaved: [0, 0.5, -0.5, 2, -2, .nan], channels: 2, sampleRate: 48_000)
        XCTAssertEqual(data.count, 44 + 12)
        XCTAssertEqual(String(decoding: data[0 ..< 4], as: UTF8.self), "RIFF")
        XCTAssertEqual(data[22], 2)
        XCTAssertEqual(data[24 ..< 28].withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }, 48_000)
        let pcm = data.dropFirst(44).withUnsafeBytes { raw in raw.bindMemory(to: Int16.self).map { $0 } }
        XCTAssertEqual(pcm, [0, 16383, -16383, 32767, -32767, 0])
    }
}
