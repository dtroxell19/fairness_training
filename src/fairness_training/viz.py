"""
Visualization utilities for fairness_training.

Requires matplotlib (install with: pip install fairness_training[viz])
"""

from __future__ import annotations

from typing import Dict, List, Optional


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError(
            "Visualization utilities require matplotlib. "
            "Install it with: pip install fairness_training[viz]  "
            "or: pip install matplotlib"
        )


def plot_training_history(
    history: Dict[str, List[float]],
    title: str = "Training History",
    figsize: tuple = (12, 4),
    save_path: Optional[str] = None,
):
    """Plot loss and fairness gap curves from FairTrainer.fit() history.

    Args:
        history: Dict returned by ``trainer.fit()``. Expected keys:
                 ``train_loss``, ``val_loss`` (optional),
                 ``train_fairness_gap``, ``val_fairness_gap`` (optional).
        title: Figure title.
        figsize: Matplotlib figure size ``(width, height)``.
        save_path: If provided, save the figure to this path instead of showing.

    Returns:
        ``matplotlib.figure.Figure``
    """
    plt = _require_matplotlib()

    has_val = "val_loss" in history and len(history["val_loss"]) > 0
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title)

    # Loss
    ax1.plot(epochs, history["train_loss"], label="Train loss", color="steelblue")
    if has_val:
        ax1.plot(epochs, history["val_loss"], label="Val loss", color="coral", linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Fairness gap
    if "train_fairness_gap" in history:
        ax2.plot(epochs, history["train_fairness_gap"], label="Train gap", color="steelblue")
    if has_val and "val_fairness_gap" in history:
        ax2.plot(
            range(1, len(history["val_fairness_gap"]) + 1),
            history["val_fairness_gap"],
            label="Val gap", color="coral", linestyle="--",
        )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Fairness gap")
    ax2.set_title("Fairness Gap")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()
    return fig


def plot_group_distributions(
    predictions,
    protected_indicators: Dict[int, "np.ndarray"],
    attr_names: Optional[Dict[int, str]] = None,
    title: str = "Prediction Distributions by Group",
    bins: int = 30,
    figsize: Optional[tuple] = None,
    save_path: Optional[str] = None,
):
    """Plot prediction distributions for each protected group as histograms.

    Args:
        predictions: Array of model predictions, shape ``(n,)`` or ``(n, 1)``.
        protected_indicators: Dict mapping ``attr_idx -> binary numpy array``
                              of 0/1 group membership. Typically from
                              ``trainer.evaluate(..., return_predictions=True)``
                              under the ``'protected'`` key.
        attr_names: Optional dict mapping ``attr_idx -> human-readable name``
                    (e.g. ``{0: 'Gender', 1: 'Race'}``).
        title: Figure title.
        bins: Number of histogram bins.
        figsize: Matplotlib figure size. Defaults to ``(6 * n_attrs, 4)``.
        save_path: If provided, save figure to this path.

    Returns:
        ``matplotlib.figure.Figure``
    """
    import numpy as np
    plt = _require_matplotlib()

    preds = np.asarray(predictions).flatten()
    n_attrs = len(protected_indicators)
    figsize = figsize or (6 * n_attrs, 4)
    fig, axes = plt.subplots(1, n_attrs, figsize=figsize, squeeze=False)
    fig.suptitle(title)

    for col, (attr_idx, indicators) in enumerate(protected_indicators.items()):
        ax = axes[0][col]
        name = (attr_names or {}).get(attr_idx, f"Attr {attr_idx}")
        ind = np.asarray(indicators).flatten()

        g0 = preds[ind == 0]
        g1 = preds[ind == 1]

        ax.hist(g0, bins=bins, alpha=0.6, label=f"{name}=0 (n={len(g0)})", color="steelblue")
        ax.hist(g1, bins=bins, alpha=0.6, label=f"{name}=1 (n={len(g1)})", color="coral")

        if len(g0):
            ax.axvline(g0.mean(), color="steelblue", linestyle="--", linewidth=1.5,
                       label=f"mean={g0.mean():.3f}")
        if len(g1):
            ax.axvline(g1.mean(), color="coral", linestyle="--", linewidth=1.5,
                       label=f"mean={g1.mean():.3f}")

        ax.set_xlabel("Prediction")
        ax.set_ylabel("Count")
        ax.set_title(name)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()
    return fig


def plot_fairness_tradeoff(
    results: Dict[str, Dict[str, float]],
    loss_key: str = "test_loss",
    gap_key: str = "weighted_avg_fairness_gap",
    title: str = "Fairness–Accuracy Tradeoff",
    figsize: tuple = (7, 5),
    save_path: Optional[str] = None,
):
    """Scatter plot of loss vs fairness gap for multiple model configurations.

    Useful for comparing models trained with different ``fairness_tolerance``
    values or different metrics.

    Args:
        results: Dict mapping a label string to a metrics dict (as returned
                 by ``trainer.evaluate()``). Example::

                     {
                         "ε=0.01": trainer_01.evaluate(test_loader),
                         "ε=0.05": trainer_05.evaluate(test_loader),
                         "No fairness": baseline_metrics,
                     }

        loss_key: Key in each metrics dict to use as the x-axis.
        gap_key: Key in each metrics dict to use as the y-axis.
        title: Figure title.
        figsize: Matplotlib figure size.
        save_path: If provided, save figure to this path.

    Returns:
        ``matplotlib.figure.Figure``
    """
    plt = _require_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    for label, metrics in results.items():
        x = metrics.get(loss_key)
        y = metrics.get(gap_key)
        if x is None or y is None:
            continue
        ax.scatter(x, y, s=80, zorder=3)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)

    ax.set_xlabel(loss_key.replace("_", " ").title())
    ax.set_ylabel(gap_key.replace("_", " ").title())
    ax.set_title(title)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()
    return fig
