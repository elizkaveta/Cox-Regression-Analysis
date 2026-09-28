import argparse
import math
from pathlib import Path

import numpy as np
from scipy.special import expit
import torch
from threadpoolctl import threadpool_limits

from data import load_dataset
from settings import DATASETS, REFERENCE_BETA

DELTAS = (0.2, 0.1, 0.05, 0.025, 0.0125)
RHOS = (0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003)
POINTS = {
    "half_reference": 0.5,
    "reference": 1.0,
    "one_half_reference": 1.5,
    "zero": 0.0,
}
CASES = (
    ("support2", "SUPPORT2", "#2166AC"),
    ("nwtco", "NWTCO", "#B68A00"),
    ("syn_n100000_d10_c90_ar095", "Correlated 100k", "#D66A21"),
    ("syn_n1000000_d20_c90_ar090", "Correlated 1M", "#638530"),
    ("syn_standard_n100000_d20_c30", "Independent 100k", "#BD5C8A"),
)


def profile_weights(scores, law, rho, center, kappa, tolerance=5e-13, limit=100):
    if np.ptp(scores) == 0:
        return np.ones_like(scores), 0.0
    lo = float(scores.min() + math.log1p(-rho) - center)
    hi = min(0.0, float(scores.max() + math.log1p(-rho) - center))
    if rho * kappa < 1:
        lo = max(lo, math.log1p(-rho * kappa))
    z = min(hi, max(lo, -rho * kappa))
    for _ in range(limit):
        weights = expit(math.log(rho) + scores - center - z) / rho
        residual = float(law @ weights - 1)
        if abs(residual) <= tolerance:
            return weights, abs(residual)
        if residual > 0:
            lo = z
        else:
            hi = z
        curvature = float(law @ (weights * (1 - rho * weights)))
        proposal = z + residual / curvature if curvature > 0 else math.inf
        z = proposal if math.isfinite(proposal) and lo < proposal < hi else (lo + hi) / 2
    raise RuntimeError(f"Shift profile root did not converge: rho={rho}, residual={residual}")


def compute(X, durations, events, reference, dataset, label):
    from solvers import group_boundaries

    order = np.argsort(-durations, kind="stable")
    X = np.ascontiguousarray(X[order], dtype=np.float64)
    times = durations[order]
    event_indices = np.flatnonzero(events[order])
    risk_end = np.searchsorted(-times, -times, side="right")
    sizes = risk_end[event_indices]
    n, _ = X.shape
    event_mean = X[event_indices].mean(axis=0)
    grouping, softplus = [], []
    for point, factor in POINTS.items():
        beta = factor * reference
        scores = X @ beta
        center = float(scores.max())
        exponentials = np.exp(scores - center)
        prefix = np.cumsum(exponentials)
        prefix_x = np.cumsum(X * exponentials[:, None], axis=0)
        event_means = prefix_x[sizes - 1] / prefix[sizes - 1, None]
        normalizers = prefix[sizes - 1] / sizes
        full_gradient = event_means.mean(axis=0) - event_mean + 0.001 * beta
        prefix2 = np.cumsum(exponentials**2)
        kappa = max(1.0, float(np.max(sizes * prefix2[sizes - 1] / prefix[sizes - 1] ** 2)))
        unique_sizes = np.unique(sizes)
        eligible = np.flatnonzero(unique_sizes[:-1] >= 0.5 * unique_sizes[1:]) + 1
        starts = np.r_[0, unique_sizes[:-1]]
        block_mass = np.add.reduceat(exponentials[: unique_sizes[-1]], starts)
        enrichments = (
            unique_sizes[eligible]
            / (unique_sizes[eligible] - unique_sizes[eligible - 1])
            * block_mass[eligible]
            / prefix[unique_sizes[eligible] - 1]
        )
        Lq = max(1.0, float(enrichments.max())) if len(eligible) else 1.0
        common = dict(
            dataset=dataset,
            label=label,
            point=point,
            reference_factor=factor,
            beta_norm=float(np.linalg.norm(beta)),
            kappa_pointwise=kappa,
            L_q_pointwise=Lq,
        )

        def grouped_gradient(groups):
            masses = np.add.reduceat(normalizers, groups.starts)
            numerator = np.add.reduceat(event_means * normalizers[:, None], groups.starts, axis=0)
            return groups.probabilities @ (numerator / masses[:, None]) - event_mean + 0.001 * beta

        for delta in DELTAS:
            groups = group_boundaries(event_indices, risk_end, delta)
            r = delta / (1 + delta)
            gradient = grouped_gradient(groups)
            grouping.append(
                dict(
                    **common,
                    delta=delta,
                    r_delta=r,
                    K=len(groups.starts),
                    score_error=float(np.linalg.norm(gradient - full_gradient)),
                    pointwise_grouping_condition=r <= 0.5 and Lq * r <= 0.5,
                )
            )
        groups = group_boundaries(event_indices, risk_end, 0.05)
        exact_gradient = grouped_gradient(groups)
        for rho in RHOS:
            mass = np.zeros(n)
            residuals = []
            for start, end, probability in zip(groups.starts, groups.ends, groups.probabilities):
                local_sizes = sizes[start:end]
                support = int(local_sizes[-1])
                bins = np.bincount(local_sizes - 1, weights=1 / local_sizes, minlength=support)
                law = np.cumsum(bins[::-1])[::-1] / (end - start)
                local_scores = scores[:support]
                local_center = float(np.log(normalizers[start:end].mean()) + center)
                W = np.exp(local_scores - local_center)
                weights, residual = profile_weights(
                    local_scores, law, rho, local_center, float(law @ (W * W))
                )
                mass[:support] += probability * (law * weights)
                residuals.append(residual)
            gradient = X.T @ mass - event_mean + 0.001 * beta
            softplus.append(
                dict(
                    **common,
                    rho=rho,
                    delta=0.05,
                    K=len(groups.starts),
                    score_error=float(np.linalg.norm(gradient - exact_gradient)),
                    pointwise_compression_conditions=Lq * (0.05 / 1.05) <= 0.5
                    and rho * kappa <= 1 / 9,
                    max_shift_residual=max(residuals),
                )
            )
    return grouping, softplus


def render(grouping, softplus, output, cases=CASES):
    from matplotlib.lines import Line2D
    from matplotlib.ticker import NullLocator

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    specs = (
        (grouping, "delta", "pointwise_grouping_condition"),
        (softplus, "rho", "pointwise_compression_conditions"),
    )
    plt.rcdefaults()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 8.7,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.7,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))
    fig.subplots_adjust(left=0.086, right=0.985, bottom=0.20, top=0.72, wspace=0.30)
    slopes = {}
    groups = {}
    for ax, (rows, xkey, condition) in zip(axes, specs):
        for dataset, label, color in cases:
            points = sorted(
                [r for r in rows if r["dataset"] == dataset and r["point"] == "reference"],
                key=lambda r: float(r[xkey]),
            )
            x = np.array([float(r[xkey]) for r in points])
            y = np.array([float(r["score_error"]) for r in points])
            if not np.all((y > 0) & np.isfinite(y)):
                raise ValueError(f"Invalid diagnostic gradient errors: {xkey}, {dataset}")
            ax.plot(x, y, color=color, lw=1.3)
            for xx, yy, row in zip(x, y, points):
                filled = row[condition]
                ax.plot(
                    xx,
                    yy,
                    marker="o",
                    ms=3.8,
                    color=color,
                    mew=0.9,
                    mfc=color if filled else "white",
                    zorder=4,
                )
            slopes[dataset, xkey] = np.polyfit(np.log(x), np.log(y), 1)[0]
            if xkey == "delta":
                groups[dataset] = next(int(r["K"]) for r in points if float(r[xkey]) == 0.05)
    guides = (
        (
            np.array([0.0125, 0.2]),
            lambda x: 2e-6 * (x / 0.0125) ** 2,
            r"$\propto\delta^2$",
        ),
        (np.array([3e-5, 0.01]), lambda x: 2e-5 * x / 3e-5, r"$\propto\rho$"),
    )
    for ax, (x, fn, label) in zip(axes, guides):
        ax.plot(x, fn(x), color="#666666", lw=1, ls="--", zorder=1)
        ax.set(xscale="log", yscale="log")
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.55)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.text(
            0.96,
            0.04,
            label,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#555555",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1),
        )
    axes[0].set(
        title="(a) Grouping",
        xlabel=r"Grouping tolerance $\delta$",
        ylabel="Gradient error",
    )
    axes[0].set_xticks([0.0125, 0.05, 0.2], ["0.0125", "0.05", "0.2"])
    axes[1].set(
        title="(b) Softplus",
        xlabel=r"Softplus parameter $\rho$",
        ylabel="Gradient error",
    )
    axes[1].set_xticks([1e-4, 1e-3, 1e-2])
    handles = [
        Line2D([0], [0], color=color, marker="o", ms=3.6, lw=1.3, label=label)
        for _, label, color in cases
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=3,
        frameon=False,
        fontsize=8,
        columnspacing=1.2,
        handlelength=1.8,
        labelspacing=0.5,
    )
    (output / "figures").mkdir(parents=True, exist_ok=True)
    (output / "tables").mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output / "figures" / "theory_diagnostics.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)
    body = [
        " & ".join((label, str(groups[d]), f"{slopes[d, 'delta']:.3f}", f"{slopes[d, 'rho']:.3f}"))
        + r" \\"
        for d, label, _ in cases
    ]
    content = [
        r"\begin{table}[htbp]",
        r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Group count at $\delta=0.05$ and descriptive log--log slopes at $\beta_{\rm ref}$, using every point in each parameter grid.}",
        r"\label{tab:study27-slopes}",
        r"\begin{tabular}{lrrr}",
        r"\hline",
        r"Dataset & $K$ & $\delta$ slope & $\rho$ slope \\",
        r"\hline",
        *body,
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    (output / "tables" / "diagnostic_slopes.tex").write_text("\n".join(content))


def save_rows(path, rows):
    np.savez_compressed(path, **{key: np.asarray([row[key] for row in rows]) for key in rows[0]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("all", *DATASETS), default="all")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("diagnostics"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = [case for case in CASES if args.dataset in ("all", case[0])]
    grouping, softplus = [], []
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        for dataset, label, _ in cases:
            print(f"{label}: computing approximation diagnostics", flush=True)
            train, _, _ = load_dataset(dataset, args.data_dir, args.download)
            rows = compute(
                train.X.numpy(),
                train.durations.numpy(),
                train.events.numpy(),
                np.asarray(REFERENCE_BETA[dataset]),
                dataset,
                label,
            )
            grouping.extend(rows[0])
            softplus.extend(rows[1])
    save_rows(args.output / "grouping_errors.npz", grouping)
    save_rows(args.output / "softplus_errors.npz", softplus)
    render(grouping, softplus, args.output, cases)
    print(f"Saved diagnostics to {args.output}")


if __name__ == "__main__":
    main()
