// YuE2Licence.swift — the offer's licence fields, decided by the operator (AB-D-0073 reversal 2026-10-24; A-6 applied
// 2026-10-24: "Declare the YuE2 offer commercial_use: conditional per AUDIO-EVALUATION A-6, with the individual-creator
// condition in the notice"). The engine package (V3) and any MES adapter read these; nothing else in the fleet restates them.
//
// Facts behind the strings (both texts ship in every bundle):
//   • HF `LICENSE` on YuE2-3B / YuE2-Vae / YuE2-Vae-legacy: plain CC BY-NC 4.0 + a scope/attribution header.
//   • GitHub `MODEL_LICENSE` (multimodal-art-projection/YuE@1647252): adds "Individual creator permission and academic use
//     (2026-09-16)" — individuals acting in an individual capacity may generate, publish, sell and license outputs with no
//     commercial licence; companies may not use the weights commercially; weight redistribution stays CC BY-NC.
public enum YuE2Licence {
    public static let spdx = "CC-BY-NC-4.0"
    public static let termsURL = "https://huggingface.co/m-a-p/YuE2-3B/blob/main/LICENSE"
    public static let modelLicenseURL = "https://github.com/multimodal-art-projection/YuE/blob/1647252/MODEL_LICENSE"
    /// A-6: commercial use is allowed under a condition the engine cannot verify (the operator is an individual).
    public static let commercialUse = "conditional"
    public static let acceptedRequired = true
    public static let namedOnly = true
    public static let notice =
        "YuE2-3B by M-A-P (huggingface.co/m-a-p/YuE2-3B), CC BY-NC 4.0 with M-A-P's individual-creator permission "
        + "(MODEL_LICENSE, 2026-09-16). Individuals acting in a personal capacity — personal users, content creators, "
        + "musicians — may publish, sell and license songs made with it. Companies may not use the model commercially "
        + "without a licence from M-A-P (ryuanab@connect.ust.hk); a hosted service making songs for paying users is "
        + "commercial use. Responsible use is a condition of the permission."
    public static let notes =
        "Covers and supplied lyrics may need the original song's rights; this model's licence doesn't grant them."
    /// The attribution the weights' licence asks for when the weights are shared.
    public static let attribution = "YuE2, YuE2-3B / YuE2-Vae / YuE2-Vae-legacy, https://huggingface.co/m-a-p/YuE2-3B"
}
