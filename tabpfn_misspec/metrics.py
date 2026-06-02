"""Evaluation metrics: C2ST and MMD."""

from typing import Optional
import numpy as np
import torch
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier


def c2st(
    X: torch.Tensor,
    Y: torch.Tensor,
    seed: int = 1,
    n_folds: int = 5,
    scoring: str = "accuracy",
    z_score: bool = True,
    noise_scale: Optional[float] = None,
) -> torch.Tensor:
    """Classifier-based 2-sample test returning accuracy

    Trains classifiers with N-fold cross-validation [1]. Scikit learn MLPClassifier are
    used, with 2 hidden layers of 10x dim each, where dim is the dimensionality of the
    samples X and Y.

    Args:
        X: Sample 1
        Y: Sample 2
        seed: Seed for sklearn
        n_folds: Number of folds
        z_score: Z-scoring using X
        noise_scale: If passed, will add Gaussian noise with std noise_scale to samples

    References:
        [1]: https://scikit-learn.org/stable/modules/cross_validation.html
    """
    if z_score:
        X_mean = torch.mean(X, dim=0)
        X_std = torch.std(X, dim=0)
        X = (X - X_mean) / X_std
        Y = (Y - X_mean) / X_std

    if noise_scale is not None:
        X += noise_scale * torch.randn(X.shape)
        Y += noise_scale * torch.randn(Y.shape)

    X = X.cpu().numpy()
    Y = Y.cpu().numpy()

    ndim = X.shape[1]

    clf = MLPClassifier(
        activation="relu",
        hidden_layer_sizes=(10 * ndim, 10 * ndim),
        max_iter=10000,
        solver="adam",
        random_state=seed,
    )

    data = np.concatenate((X, Y))
    target = np.concatenate(
        (
            np.zeros((X.shape[0],)),
            np.ones((Y.shape[0],)),
        )
    )

    shuffle = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, data, target, cv=shuffle, scoring=scoring)

    scores = np.asarray(np.mean(scores)).astype(np.float32)
    return torch.from_numpy(np.atleast_1d(scores))


def mmd(X, Y, alpha: float = 1.0):
    """Maximum mean discrepancy with rational quadratic kernel (median heuristic).

    Uses the rational quadratic kernel
    ``k(x, y) = (1 + ||x - y||^2 / (2 * alpha * l^2)) ** (-alpha)``,
    a scale mixture of RBF kernels that recovers the Gaussian kernel as
    ``alpha -> inf``. The length scale ``l`` is set by the median heuristic.

    Args:
        X: Tensor or array of shape (N, D).
        Y: Tensor or array of shape (M, D).
        alpha: Rational quadratic mixture parameter (> 0).

    Returns:
        MMD^2 estimate (float).
    """
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float()
    if isinstance(Y, np.ndarray):
        Y = torch.from_numpy(Y).float()

    X = X.detach().float()
    Y = Y.detach().float()

    # Median heuristic for the length scale l (l^2 = median squared distance)
    XY = torch.cat([X, Y], dim=0)
    dists = torch.cdist(XY, XY)
    median_dist = dists[dists > 0].median()
    denom = 2.0 * alpha * median_dist**2

    def rq(a, b):
        return (1.0 + torch.cdist(a, b) ** 2 / denom) ** (-alpha)

    K_XX = rq(X, X)
    K_YY = rq(Y, Y)
    K_XY = rq(X, Y)

    n = X.shape[0]
    m = Y.shape[0]

    mmd_sq = (
        (K_XX.sum() - K_XX.trace()) / (n * (n - 1))
        + (K_YY.sum() - K_YY.trace()) / (m * (m - 1))
        - 2.0 * K_XY.sum() / (n * m)
    )
    return float(mmd_sq)
