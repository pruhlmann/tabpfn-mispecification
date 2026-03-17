"""Compare sklearn C2ST vs PyTorch C2ST on saved posterior samples."""

import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier


# ── sklearn version (current) ────────────────────────────────────────────────

def c2st_sklearn(
    X: torch.Tensor,
    Y: torch.Tensor,
    seed: int = 1,
    n_folds: int = 5,
    z_score: bool = True,
    noise_scale: Optional[float] = None,
) -> float:
    if z_score:
        X_mean = torch.mean(X, dim=0)
        X_std = torch.std(X, dim=0)
        X = (X - X_mean) / X_std
        Y = (Y - X_mean) / X_std
    if noise_scale is not None:
        X += noise_scale * torch.randn(X.shape)
        Y += noise_scale * torch.randn(Y.shape)

    X_np = X.cpu().numpy()
    Y_np = Y.cpu().numpy()
    ndim = X_np.shape[1]

    clf = MLPClassifier(
        activation="relu",
        hidden_layer_sizes=(10 * ndim, 10 * ndim),
        max_iter=10000,
        solver="adam",
        random_state=seed,
    )
    data = np.concatenate((X_np, Y_np))
    target = np.concatenate((np.zeros(X_np.shape[0]), np.ones(Y_np.shape[0])))
    shuffle = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, data, target, cv=shuffle, scoring="accuracy")
    return float(np.mean(scores))


# ── PyTorch version (per-fold, GPU-optimized) ───────────────────────────────

class _C2ST_MLP(nn.Module):
    def __init__(self, ndim: int):
        super().__init__()
        h = 10 * ndim
        self.net = nn.Sequential(
            nn.Linear(ndim, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _kfold_indices(n: int, n_splits: int, seed: int):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(kf.split(np.arange(n)))


@torch.no_grad()
def _eval_acc(model, x, y):
    model.eval()
    preds = (model(x) > 0).float()
    return (preds == y).float().mean().item()


def _train_fold(
    x_train, y_train, x_test, y_test, ndim, seed, device,
    max_iter=10000, lr=0.001, alpha=0.0001, tol=0.0001, n_iter_no_change=10,
):
    """Train one fold. Uses full-batch on GPU, mini-batch on CPU."""
    torch.manual_seed(seed)
    model = _C2ST_MLP(ndim).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=alpha
    )

    n_train = len(x_train)
    use_minibatch = device.type == "cpu"
    batch_size = min(200, n_train) if use_minibatch else n_train

    best_loss = float("inf")
    no_improve = 0

    for epoch in range(max_iter):
        model.train()
        if use_minibatch:
            perm = torch.randperm(n_train, device=device)
            x_s = x_train[perm]
            y_s = y_train[perm]
        else:
            # Full-batch on GPU: shuffle once via permutation
            perm = torch.randperm(n_train, device=device)
            x_s = x_train[perm]
            y_s = y_train[perm]

        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n_train, batch_size):
            xb = x_s[i : i + batch_size]
            yb = y_s[i : i + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        if best_loss - avg_loss > tol:
            best_loss = avg_loss
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= n_iter_no_change:
            break

    return _eval_acc(model, x_test, y_test)


def c2st_pytorch(
    X: torch.Tensor,
    Y: torch.Tensor,
    seed: int = 1,
    n_folds: int = 5,
    z_score: bool = True,
    noise_scale: Optional[float] = None,
) -> float:
    """PyTorch C2ST matching sklearn MLPClassifier parameters."""
    device = X.device

    if z_score:
        X_mean = torch.mean(X, dim=0)
        X_std = torch.std(X, dim=0)
        X = (X - X_mean) / X_std
        Y = (Y - X_mean) / X_std
    if noise_scale is not None:
        X = X + noise_scale * torch.randn(X.shape, device=device)
        Y = Y + noise_scale * torch.randn(Y.shape, device=device)

    ndim = X.shape[1]
    data = torch.cat([X, Y], dim=0).float().to(device)
    target = torch.cat([
        torch.zeros(X.shape[0], device=device),
        torch.ones(Y.shape[0], device=device),
    ]).float()

    folds = _kfold_indices(len(data), n_folds, seed)
    scores = []
    for fold_i, (train_idx, test_idx) in enumerate(folds):
        train_idx_t = torch.from_numpy(train_idx).long().to(device)
        test_idx_t = torch.from_numpy(test_idx).long().to(device)
        acc = _train_fold(
            data[train_idx_t], target[train_idx_t],
            data[test_idx_t], target[test_idx_t],
            ndim, seed + fold_i, device,
        )
        scores.append(acc)
    return float(np.mean(scores))


# ── Test runner ──────────────────────────────────────────────────────────────

def test_on_saved_samples(posterior_path, reference_path, label=""):
    post = torch.load(posterior_path, weights_only=True, map_location="cpu")
    ref = torch.load(reference_path, weights_only=True, map_location="cpu")
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  posterior: {post.shape}, reference: {ref.shape}")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    sk_val = c2st_sklearn(post, ref)
    t_sk = time.perf_counter() - t0

    t0 = time.perf_counter()
    pt_cpu_val = c2st_pytorch(post, ref)
    t_pt_cpu = time.perf_counter() - t0

    if torch.cuda.is_available():
        # Warmup
        _ = c2st_pytorch(post[:100].cuda(), ref[:100].cuda())
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        pt_gpu_val = c2st_pytorch(post.cuda(), ref.cuda())
        torch.cuda.synchronize()
        t_pt_gpu = time.perf_counter() - t0
    else:
        pt_gpu_val = float("nan")
        t_pt_gpu = float("nan")

    print(f"  sklearn:     {sk_val:.4f}  ({t_sk:.2f}s)")
    print(f"  pytorch-cpu: {pt_cpu_val:.4f}  ({t_pt_cpu:.2f}s)")
    print(f"  pytorch-gpu: {pt_gpu_val:.4f}  ({t_pt_gpu:.2f}s)")
    print(f"  diff (sk-pt_cpu): {abs(sk_val - pt_cpu_val):.4f}")
    if not np.isnan(pt_gpu_val):
        print(f"  diff (sk-pt_gpu): {abs(sk_val - pt_gpu_val):.4f}")
    return sk_val, pt_cpu_val, pt_gpu_val


def test_known_distributions():
    print(f"\n{'='*60}")
    print("  Synthetic tests")
    print(f"{'='*60}")

    torch.manual_seed(42)

    X = torch.randn(500, 2)
    Y = torch.randn(500, 2)
    sk = c2st_sklearn(X, Y)
    pt = c2st_pytorch(X, Y)
    print(f"  Same dist:   sklearn={sk:.4f}  pytorch={pt:.4f}  (expect ~0.5)")

    X = torch.randn(500, 2)
    Y = torch.randn(500, 2) + 5.0
    sk = c2st_sklearn(X, Y)
    pt = c2st_pytorch(X, Y)
    print(f"  Diff dist:   sklearn={sk:.4f}  pytorch={pt:.4f}  (expect ~1.0)")

    X = torch.randn(500, 2)
    Y = torch.randn(500, 2) + 0.5
    sk = c2st_sklearn(X, Y)
    pt = c2st_pytorch(X, Y)
    print(f"  Slight diff: sklearn={sk:.4f}  pytorch={pt:.4f}  (expect ~0.6-0.8)")


if __name__ == "__main__":
    from pathlib import Path

    test_known_distributions()

    results_dir = Path("results")
    test_cases = [
        ("two_moons_heavy_tail_radius/artifacts/ncalib50_seed42",
         "npepfn_misspec_seed42_obs1.pt", "reference_seed42_obs1.pt",
         "two_moons heavy_tail npepfn_misspec ncalib50"),
        ("two_moons_heavy_tail_radius/artifacts/ncalib50_seed42",
         "npepfn_y_fmpe_seed42_obs1.pt", "reference_seed42_obs1.pt",
         "two_moons heavy_tail npepfn_y_fmpe ncalib50"),
        ("gaussian_mixture_one_gaussian/artifacts/ncalib50_seed42",
         "npepfn_misspec_seed42_obs1.pt", "reference_seed42_obs1.pt",
         "gaussian_mixture npepfn_misspec ncalib50"),
        ("sir_weekend_delay/artifacts/ncalib50_seed42",
         "npepfn_misspec_seed42_obs1.pt", "reference_seed42_obs1.pt",
         "sir npepfn_misspec ncalib50"),
        ("slcp_diagonal_covariance/artifacts/ncalib50_seed42",
         "npepfn_misspec_seed42_obs1.pt", "reference_seed42_obs1.pt",
         "slcp npepfn_misspec ncalib50"),
    ]

    all_diffs_cpu = []
    all_diffs_gpu = []
    for subdir, post_file, ref_file, label in test_cases:
        post_path = results_dir / subdir / post_file
        ref_path = results_dir / subdir / ref_file
        if post_path.exists() and ref_path.exists():
            sk, pt_cpu, pt_gpu = test_on_saved_samples(post_path, ref_path, label)
            all_diffs_cpu.append(abs(sk - pt_cpu))
            if not np.isnan(pt_gpu):
                all_diffs_gpu.append(abs(sk - pt_gpu))
        else:
            print(f"\n  SKIPPED: {label} (files not found)")

    if all_diffs_cpu:
        print(f"\n{'='*60}")
        print(f"  CPU: mean |diff| = {np.mean(all_diffs_cpu):.4f}, max = {np.max(all_diffs_cpu):.4f}")
        if all_diffs_gpu:
            print(f"  GPU: mean |diff| = {np.mean(all_diffs_gpu):.4f}, max = {np.max(all_diffs_gpu):.4f}")
        print(f"{'='*60}")
