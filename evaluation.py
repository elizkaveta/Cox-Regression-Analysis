import numpy as np
import torch

try:
    from numba import njit
except ImportError:
    njit = None


def _fenwick_kernel(order, times, events, ranks, lower_bounds, upper_bounds, n_unique):
    tree = np.zeros(n_unique + 1, dtype=np.int64)
    total, concordant, tied, pairs, start = 0, 0, 0, 0, 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and times[order[stop]] == times[order[start]]:
            stop += 1

        for pos in range(start, stop):
            i = order[pos]
            if not events[i]:
                rank = ranks[i]
                while rank < len(tree):
                    tree[rank] += 1
                    rank += rank & -rank
                total += 1
        for pos in range(start, stop):
            i = order[pos]
            if events[i]:
                lower = lower_bounds[i]
                upper = upper_bounds[i]
                lo_count, hi_count = 0, 0
                while lower > 0:
                    lo_count += tree[lower]
                    lower -= lower & -lower
                while upper > 0:
                    hi_count += tree[upper]
                    upper -= upper & -upper
                concordant += lo_count
                tied += hi_count - lo_count
                pairs += total
        for pos in range(start, stop):
            i = order[pos]
            if events[i]:
                rank = ranks[i]
                while rank < len(tree):
                    tree[rank] += 1
                    rank += rank & -rank
                total += 1
        start = stop
    return (concordant + 0.5 * tied) / pairs if pairs else np.nan


def _tie_search_bounds(unique, risks, tie_tol):
    with np.errstate(over="ignore"):
        lower = np.ascontiguousarray(
            np.searchsorted(unique, risks - tie_tol, side="left"), dtype=np.int64
        )
        upper = np.ascontiguousarray(
            np.searchsorted(unique, risks + tie_tol, side="right"), dtype=np.int64
        )
        size = len(unique)
        if not size:
            return lower, upper
        lower_bad = ((lower > 0) & (risks - unique[np.maximum(lower - 1, 0)] <= tie_tol)) | (
            (lower < size) & (risks - unique[np.minimum(lower, size - 1)] > tie_tol)
        )
        upper_bad = ((upper < size) & (unique[np.minimum(upper, size - 1)] - risks <= tie_tol)) | (
            (upper > 0) & (unique[np.maximum(upper - 1, 0)] - risks > tie_tol)
        )


        for bounds, bad, is_lower in ((lower, lower_bad, True), (upper, upper_bad, False)):
            indices = np.flatnonzero(bad)
            if not len(indices):
                continue
            selected = risks[indices]
            lo = np.zeros(len(indices), dtype=np.int64)
            hi = np.full(len(indices), size, dtype=np.int64)
            while (lo < hi).any():
                active = lo < hi
                midpoint = (lo + hi) // 2
                candidate = unique[np.minimum(midpoint, size - 1)]
                move_left = (
                    (selected - candidate <= tie_tol)
                    if is_lower
                    else (candidate - selected > tie_tol)
                )
                hi = np.where(active & move_left, midpoint, hi)
                lo = np.where(active & ~move_left, midpoint + 1, lo)
            bounds[indices] = lo
        return lower, upper


def harrell_cindex(durations, events, scores, tied_tol=1e-8):
    def array(x, dtype):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return np.ascontiguousarray(x, dtype=dtype)

    times = array(durations, np.float64)
    ev = array(events, np.bool_).view(np.uint8)
    risk = array(scores, np.float64)
    if times.ndim != 1 or times.shape != ev.shape or times.shape != risk.shape:
        raise ValueError("C-index arrays must have equal one-dimensional shapes")
    if (
        not np.isfinite(times).all()
        or not np.isfinite(risk).all()
        or not np.isfinite(tied_tol)
        or tied_tol < 0
    ):
        raise ValueError("C-index needs finite inputs and nonnegative tie tolerance")
    return float(_fenwick_concordance(times, ev, risk, tied_tol))


if njit is not None:
    _fenwick_kernel = njit(cache=True)(_fenwick_kernel)


def _fenwick_concordance(times, events, risks, tie_tol):
    order = np.ascontiguousarray(np.argsort(times, kind="stable")[::-1], dtype=np.int64)
    unique, inverse = np.unique(risks, return_inverse=True)
    ranks = np.ascontiguousarray(inverse + 1, dtype=np.int64)
    lower, upper = _tie_search_bounds(unique, risks, tie_tol)
    return _fenwick_kernel(order, times, events, ranks, lower, upper, len(unique))


def warm_evaluator():
    harrell_cindex([1.0, 2.0], [1, 0], [1.0, 0.0])


def full_cox_loss_from_sorted_scores(data, scores):
    if data.n_events == 0:
        raise ValueError("Cox loss is undefined for a split without events")
    log_risk = torch.logcumsumexp(scores, dim=0)
    return (log_risk[data.risk_end[data.event_indices] - 1] - scores[data.event_indices]).mean()


@torch.no_grad()
def validation_metrics(data, beta):
    beta = torch.as_tensor(beta, dtype=torch.float64)
    scores = data.X_sorted @ beta
    return {
        "val_cox_loss": float(full_cox_loss_from_sorted_scores(data, scores)),
        "val_cindex": harrell_cindex(data.durations_sorted, data.events_sorted, scores),
    }


@torch.no_grad()
def evaluate(train, val, beta, reference, ridge_lambda=0.001):
    beta = torch.as_tensor(beta, dtype=torch.float64)
    loss = float(full_cox_loss_from_sorted_scores(train, train.X_sorted @ beta))
    penalty = float((0.5 * ridge_lambda) * beta.square().sum())
    return {
        "train_cox_loss": loss,
        "train_regularized_objective": loss + penalty,
        "train_gap": loss + penalty - reference["loss"],
        "gap_resolution": reference["gap_resolution"],
        **validation_metrics(val, beta),
    }
