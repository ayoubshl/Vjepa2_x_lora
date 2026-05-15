"""
Smoke test — verify the whole module graph works without GPU or real data.

# This catches dumb bugs (typos, import loops, shape mismatches in the
# probe construction) BEFORE you waste hours on a cluster setting up data.
#
# What it does:
#   1. Imports every module
#   2. Builds the probe with dummy dims
#   3. Runs a fake forward+backward through the probe with random tensors
#   4. Verifies the loss is differentiable
#   5. Tests mean-class R@5 computation on synthetic data
#
# What it does NOT do:
#   - Load V-JEPA 2 (needs HF + GPU)
#   - Load real video (needs Decord + data on disk)
#   - Apply LoRA to a real model (PEFT needs the model loaded)
#
# Run:  python scripts/smoke_test.py
"""

import os
import sys
import traceback

# Make `src` importable when this script is run from anywhere.
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch


def section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def main():
    print("V-JEPA 2 × LoRA smoke test")

    # ---- 1. Imports ----
    section("1. Import test")
    try:
        from src.seed import set_seed
        from src.vocabulary import build_action_vocabulary, load_action_vocabulary, get_action_id
        from src.dataset import compute_frame_indices
        from src.probe import AttentiveProbe, build_probe
        from src.losses import FocalLoss, MultiHeadLoss, build_loss
        from src.evaluate import mean_class_recall_at_k
        from src.monitor import CollapseMonitor
        print("OK — all src modules import")
    except ImportError as e:
        print(f"FAIL: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ---- 2. Seed ----
    section("2. Seed")
    set_seed(42, deterministic=False)
    print("OK")

    # ---- 3. Frame index computation ----
    section("3. compute_frame_indices")
    # Valid case
    idx = compute_frame_indices(start_frame=600, fps_source=60, fps_target=8,
                                 num_frames=32, anticipation_seconds=1.0)
    assert idx is not None and len(idx) == 32, f"expected 32 indices, got {len(idx) if idx is not None else None}"
    assert idx[0] == 600 - 60 - 240 and idx[-1] == 600 - 60 - 1, \
           f"indices wrong: first={idx[0]}, last={idx[-1]} (expected 300, 539)"
    print(f"OK — 32 indices from {idx[0]} to {idx[-1]} (4-second window, 8 FPS)")

    # Underflow case
    idx = compute_frame_indices(start_frame=100, fps_source=60, fps_target=8,
                                 num_frames=32, anticipation_seconds=1.0)
    assert idx is None, f"expected None for underflow case, got {idx}"
    print("OK — underflow correctly returns None")

    # ---- 4. Probe forward + backward ----
    section("4. AttentiveProbe forward+backward")
    enc_dim, pred_dim = 1024, 384
    num_verb, num_noun, num_action = 97, 300, 3000
    probe = build_probe(
        encoder_dim=enc_dim,
        predictor_dim=pred_dim,
        num_action_classes=num_action,
        num_verb_classes=num_verb,
        num_noun_classes=num_noun,
        depth=4, num_heads=16, mlp_ratio=4.0, dropout=0.0,
    )
    # Note: real V-JEPA 2 ViT-L outputs roughly N_enc=N_pred=8192 tokens for
    # 32 frames at 256x256 (tubelet 2, patch 16: (32/2)*(256/16)*(256/16) =
    # 16*16*16 = 4096 — actually 4096 not 8192 unless I'm misremembering).
    # For the smoke test, use small token counts to keep it fast.
    B, N_e, N_p = 2, 64, 64
    enc = torch.randn(B, N_e, enc_dim, requires_grad=False)
    pred = torch.randn(B, N_p, pred_dim, requires_grad=False)

    v_log, n_log, a_log = probe(enc, pred)
    assert v_log.shape == (B, num_verb)
    assert n_log.shape == (B, num_noun)
    assert a_log.shape == (B, num_action)
    print(f"OK — output shapes: verb {v_log.shape}, noun {n_log.shape}, action {a_log.shape}")

    # ---- 5. Loss ----
    section("5. MultiHeadLoss")
    loss_fn = build_loss({"type": "focal", "gamma": 2.0,
                          "w_verb": 1.0, "w_noun": 1.0, "w_action": 1.0})
    v_lab = torch.randint(0, num_verb, (B,))
    n_lab = torch.randint(0, num_noun, (B,))
    a_lab = torch.randint(-1, num_action, (B,))  # include -1 to test ignore

    total, loss_dict = loss_fn(v_log, n_log, a_log, v_lab, n_lab, a_lab)
    assert torch.isfinite(total), f"loss is non-finite: {total}"
    loss_value = float(total.detach())
    total.backward()  # check gradient flow
    grads = sum(1 for p in probe.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    total_params = sum(1 for _ in probe.parameters())
    assert grads > 0, "no probe param received gradients"
    print(f"OK — loss={loss_value:.4f}, {grads}/{total_params} probe params received grad")
    print(f"   verb={loss_dict['verb_loss']:.4f} noun={loss_dict['noun_loss']:.4f} action={loss_dict['action_loss']:.4f}")

    # ---- 6. Mean-class R@5 metric ----
    section("6. mean_class_recall_at_k")
    # Construct synthetic case where the model is perfect for class 0,
    # random for class 1. Mean-class R@5 should be (1.0 + ~0.5)/2 ≈ 0.75
    # (well, with k=5 and 10 classes, random R@5 is 0.5)
    torch.manual_seed(0)
    N = 200
    labels = torch.cat([torch.zeros(100, dtype=torch.long),
                        torch.ones(100, dtype=torch.long)])
    logits = torch.randn(N, 10)
    # Make class 0 always rank 1 for samples labeled 0
    logits[:100, 0] = 100.0
    result = mean_class_recall_at_k(logits, labels, k=5)
    assert result["num_classes_used"] == 2
    assert result["per_class_recall"][0] == 1.0, "class 0 should be perfect"
    print(f"OK — mean R@5 = {result['mean_class_recall']:.3f} "
          f"(class 0: {result['per_class_recall'][0]:.3f}, "
          f"class 1: {result['per_class_recall'][1]:.3f})")

    # ---- 7. Ignore-index behavior in mR@5 ----
    section("7. mR@5 with ignore_index=-1")
    labels_with_neg = torch.tensor([0, 0, 1, 1, -1, -1])
    logits = torch.randn(6, 3)
    logits[:2, 0] = 10.0   # samples 0,1 predict class 0 (correct, label 0)
    logits[2:4, 1] = 10.0  # samples 2,3 predict class 1 (correct, label 1)
    result = mean_class_recall_at_k(logits, labels_with_neg, k=2, ignore_index=-1)
    assert result["num_samples_used"] == 4, f"expected 4, got {result['num_samples_used']}"
    assert result["num_classes_used"] == 2
    print(f"OK — ignored 2 samples with label -1; "
          f"evaluated {result['num_samples_used']} samples / "
          f"{result['num_classes_used']} classes")

    # ---- 8. Collapse monitor ----
    section("8. CollapseMonitor")
    mon = CollapseMonitor(log_every=1)
    feats = torch.randn(16, 1024)
    out = mon.update(feats)
    assert out is not None
    assert "collapse/variance" in out
    assert "collapse/cosine_sim" in out
    assert "collapse/effective_rank" in out
    print(f"OK — variance={out['collapse/variance']:.4f}, "
          f"cosine_sim={out['collapse/cosine_sim']:.4f}, "
          f"eff_rank={out['collapse/effective_rank']:.1f}")

    # Test collapse detection: identical features → low variance, high cos sim
    collapsed = torch.ones(16, 1024) + 0.001 * torch.randn(16, 1024)
    mon2 = CollapseMonitor(log_every=1)
    out2 = mon2.update(collapsed)
    assert out2["collapse/variance"] < 0.01, "should detect collapse via variance"
    assert out2["collapse/cosine_sim"] > 0.99, "should detect collapse via cosine"
    print(f"OK — collapse detected: var={out2['collapse/variance']:.6f}, "
          f"cos={out2['collapse/cosine_sim']:.4f}")

    section("ALL SMOKE TESTS PASSED")
    print("\nNext steps:")
    print("  1. Install requirements.txt on your training machine")
    print("  2. Download EK-100 videos + annotations")
    print("  3. Run scripts/inspect_model.py to verify LoRA target_modules names")
    print("  4. (Recommended) Run scripts/verify_paper_match.py with released probe")
    print("  5. Edit configs/global.yaml paths, set participants.txt")
    print("  6. python run.py --config configs/baseline_frozen.yaml")


if __name__ == "__main__":
    main()
