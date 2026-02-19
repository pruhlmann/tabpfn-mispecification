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


def mmd(X, Y):
    """Maximum mean discrepancy with Gaussian kernel (median heuristic).

    Args:
        X: Tensor or array of shape (N, D).
        Y: Tensor or array of shape (M, D).

    Returns:
        MMD^2 estimate (float).
    """
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float()
    if isinstance(Y, np.ndarray):
        Y = torch.from_numpy(Y).float()

    X = X.detach().float()
    Y = Y.detach().float()

    # Median heuristic for bandwidth
    XY = torch.cat([X, Y], dim=0)
    dists = torch.cdist(XY, XY)
    median_dist = dists[dists > 0].median()
    gamma = 1.0 / (2.0 * median_dist**2)

    K_XX = torch.exp(-gamma * torch.cdist(X, X) ** 2)
    K_YY = torch.exp(-gamma * torch.cdist(Y, Y) ** 2)
    K_XY = torch.exp(-gamma * torch.cdist(X, Y) ** 2)

    n = X.shape[0]
    m = Y.shape[0]

    mmd_sq = (
        (K_XX.sum() - K_XX.trace()) / (n * (n - 1))
        + (K_YY.sum() - K_YY.trace()) / (m * (m - 1))
        - 2.0 * K_XY.sum() / (n * m)
    )
    return float(mmd_sq)
