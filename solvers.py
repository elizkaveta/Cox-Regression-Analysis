from dataclasses import dataclass
import math

import numpy as np
import torch
from torch.nn import functional as F


METHODS = ("rho_uniform", "batch_lse", "minibatch", "bigsurv_normalized", "cox_cc")


@dataclass
class Groups:
    starts: np.ndarray
    ends: np.ndarray
    probabilities: np.ndarray


def group_boundaries(event_indices, risk_end, delta):
    if delta <= 0 or len(event_indices) == 0:
        raise ValueError("Require positive delta and at least one event")
    starts, ends = [], []
    stop = len(event_indices)
    while stop:
        largest = int(risk_end[event_indices[stop - 1]])
        low, high = 0, stop - 1
        while low < high:
            middle = (low + high) // 2
            size = int(risk_end[event_indices[middle]])
            if largest <= (1 + delta) * size:
                high = middle
            else:
                low = middle + 1
        starts.append(low)
        ends.append(stop)
        stop = low
    starts = np.asarray(starts[::-1], dtype=np.int64)
    ends = np.asarray(ends[::-1], dtype=np.int64)
    return Groups(starts, ends, (ends - starts) / len(event_indices))


class UniformSoftplus:

    def __init__(self, data, config, seed, cap):
        self.X = data.X_sorted.numpy()
        self.events = data.event_indices.numpy()
        self.risk_end = data.risk_end.numpy()
        self.config, self.cap = config, cap
        self.batch = int(config["batch_size"])
        if self.batch < 1 or not 0 < config["rho"] < 1:
            raise ValueError("Require positive batch_size and 0 < rho < 1")
        self.groups = group_boundaries(self.events, self.risk_end, config["delta"])
        self.q = np.full(len(self.groups.starts), 1.0 / len(self.groups.starts))
        self.cdf = np.cumsum(self.q)
        self.cdf[-1] = 1.0
        self.beta = np.zeros(data.d)
        self.average = np.zeros_like(self.beta)
        self.shifts = np.full(len(self.q), math.log1p(-config["rho"]))
        self.rng = np.random.default_rng(seed)
        self.log_rho = math.log(config["rho"])
        self.steps = self.projections = self.units = 0

    def can_step(self):

        return self.units + 3 * self.batch + 8 <= self.cap

    def step(self):
        group_ids = np.searchsorted(self.cdf, self.rng.random(self.batch), side="right")
        positions = self.groups.starts[group_ids] + (
            self.rng.random(self.batch)
            * (self.groups.ends[group_ids] - self.groups.starts[group_ids])
        ).astype(np.int64)
        events = self.events[positions]
        subjects = (self.rng.random(self.batch) * self.risk_end[events]).astype(np.int64)
        event_x, risk_x = self.X[events], self.X[subjects]
        scores = risk_x @ self.beta
        weights = np.exp(-np.logaddexp(self.log_rho, self.shifts[group_ids] - scores))
        importance = self.groups.probabilities[group_ids] / (self.batch * self.q[group_ids])
        gradient = risk_x.T @ (importance * weights) - event_x.T @ importance
        eta = self.config["learning_rate"] / math.sqrt(1 + self.steps / self.config["decay_steps"])
        self.beta *= 1 - eta * self.config["ridge_lambda"]
        self.beta -= eta * gradient
        np.add.at(self.shifts, group_ids, eta * (weights - 1) / (self.batch * self.q[group_ids]))
        touched = np.unique(group_ids)
        self.shifts[touched] = np.clip(
            self.shifts[touched], -self.config["shift_bound"], self.config["shift_bound"]
        )
        norm = float(np.linalg.norm(self.beta))
        self.units += 3 * self.batch + 7
        if norm > self.config["beta_radius"]:
            self.beta *= self.config["beta_radius"] / norm
            self.projections += 1
            self.units += 1
        self.steps += 1
        self.average += (self.beta - self.average) / self.steps

    def readout(self):
        return self.average.copy()

    def row(self):
        return dict(
            step=self.steps,
            units_total=self.units,
            units_forward=self.steps * self.batch,
            units_risk_gradient=self.steps * self.batch,
            units_event_gradient=self.steps * self.batch,
            units_vector_algebra=self.steps,
            units_ridge=self.steps,
            units_update=self.steps,
            units_norm=self.steps,
            units_averaging=3 * self.steps,
            units_projection=self.projections,
        )


def _cox_loss(scores, durations, events, reduction="mean"):
    order = torch.argsort(durations, dim=-1, descending=True, stable=True)
    times = torch.gather(durations, -1, order).contiguous()
    eta = torch.gather(scores, -1, order)
    observed = torch.gather(events, -1, order).to(eta.dtype)
    end = torch.searchsorted((-times).contiguous(), (-times).contiguous(), right=True) - 1
    terms = (torch.gather(torch.logcumsumexp(eta, dim=-1), -1, end) - eta) * observed
    if reduction == "sum":
        return terms.sum(dim=-1)
    return terms.sum(dim=-1) / observed.sum(dim=-1).clamp_min(1)


class _BigSurvAMSGrad:

    def __init__(self, beta, config):
        self.beta, self.config = beta, config
        self.beta1, self.beta2 = 0.9, 0.99
        self.eps = 1e-8
        self.m = torch.zeros_like(beta)
        self.v = torch.zeros_like(beta)
        self.v_max = torch.zeros_like(beta)
        self.steps = 0

    @torch.no_grad()
    def step(self, gradient):
        self.steps += 1
        lr = self.config["learning_rate"] / self.steps ** self.config["lr_tau"]
        self.m.mul_(self.beta1).add_(gradient, alpha=1 - self.beta1)
        self.v.mul_(self.beta2).addcmul_(gradient, gradient, value=1 - self.beta2)
        torch.maximum(self.v_max, self.v, out=self.v_max)
        self.beta.addcdiv_(self.m, self.v_max.sqrt().add_(self.eps), value=-lr)


class SampledCox:

    def __init__(self, method, data, config, seed, cap):
        self.method, self.data, self.config, self.cap = method, data, config, cap
        self.beta = torch.zeros(data.d, dtype=torch.float64, requires_grad=True)
        self.rng = torch.Generator(device=data.X.device).manual_seed(seed)
        self.steps = self.score_rows = 0
        self.order, self.position = None, data.n
        if method == "bigsurv_normalized":
            self.strata_size = int(config["strata_size"])
            self.strata_batch_size = int(config["strata_batch_size"])
            self.batch_size = self.strata_size * self.strata_batch_size
            self.steps_per_epoch = data.n // self.batch_size
            if self.strata_size < 2 or self.strata_batch_size < 1 or self.steps_per_epoch < 1:
                raise ValueError("Training data must contain at least one complete strata batch")
            self.normalizer = self.strata_size * data.n_events / data.n
            self.optimizer = _BigSurvAMSGrad(self.beta, config)
            self.average = self.beta.detach().clone()
        else:
            self.optimizer = torch.optim.Adam(
                [self.beta],
                lr=config["learning_rate"],
                amsgrad=False,
                capturable=False,
                weight_decay=0.0,
            )
            if method == "minibatch":
                self.batch_size = min(int(config["batch_size"]), data.n)

    def _parts(self, steps, score_rows):
        return dict(
            forward=score_rows,
            backward=score_rows,
            auxiliary=2 * steps,
            vector_algebra=steps,
            optimizer=12 * steps,
            averaging=3 * steps if self.method == "bigsurv_normalized" else 0,
        )

    @property
    def units(self):
        return sum(self._parts(self.steps, self.score_rows).values())

    def can_step(self):
        cfg = self.config
        if self.method == "batch_lse":
            size = cfg["outer_batch_size"] * (cfg["risk_batch_size"] + 1)
        elif self.method == "cox_cc":
            size = cfg["case_batch_size"] * (cfg["n_controls"] + 1)
        elif self.method == "minibatch":
            steps_per_epoch = math.ceil(self.data.n / self.batch_size)
            size = self.batch_size
            if self.steps % steps_per_epoch == steps_per_epoch - 1:
                size = self.data.n % self.batch_size or self.batch_size
        else:
            size = self.batch_size
        return sum(self._parts(self.steps + 1, self.score_rows + size).values()) <= self.cap

    def _scores(self, X, indices):
        self.score_rows += indices.numel()
        return (X[indices.reshape(-1)] @ self.beta).reshape(indices.shape)

    def _case_control_scores(self, n_cases, n_controls):
        positions = torch.randint(self.data.n_events, (n_cases,), generator=self.rng)
        cases = self.data.event_indices[positions]
        uniforms = torch.rand((n_cases, n_controls), dtype=torch.float64, generator=self.rng)
        controls = (uniforms * self.data.risk_end[cases, None]).to(torch.long)
        indices = torch.cat((cases[:, None], controls), dim=1)
        return self._scores(self.data.X_sorted, indices)

    def _minibatch_loss(self):
        if self.position >= self.data.n:
            self.order = torch.randperm(self.data.n, generator=self.rng)
            self.position = 0
        indices = self.order[self.position : self.position + self.batch_size]
        self.position += indices.numel()
        return _cox_loss(
            self._scores(self.data.X, indices),
            self.data.durations[indices],
            self.data.events[indices],
        )

    def _bigsurv_loss(self):
        step_in_epoch = self.steps % self.steps_per_epoch
        if step_in_epoch == 0:
            self.order = torch.randperm(self.data.n, generator=self.rng)
        start = step_in_epoch * self.batch_size
        strata = self.order[start : start + self.batch_size].view(
            self.strata_batch_size, self.strata_size
        )
        return (
            _cox_loss(
                self._scores(self.data.X_sorted, strata),
                self.data.durations_sorted[strata],
                self.data.events_sorted[strata],
                reduction="sum",
            ).mean()
            / self.normalizer
        )

    def step(self):
        cfg = self.config
        if self.method != "bigsurv_normalized":
            self.optimizer.zero_grad(set_to_none=True)
        if self.method == "batch_lse":
            scores = self._case_control_scores(cfg["outer_batch_size"], cfg["risk_batch_size"])
            loss = (
                -scores[:, 0]
                + torch.logsumexp(scores[:, 1:], dim=1)
                - math.log(cfg["risk_batch_size"])
            ).mean()
        elif self.method == "cox_cc":
            scores = self._case_control_scores(cfg["case_batch_size"], cfg["n_controls"])
            differences = scores[:, 1:] - scores[:, :1]
            if cfg["n_controls"] == 1:
                loss = F.softplus(differences[:, 0]).mean()
            else:
                loss = torch.logsumexp(
                    torch.cat((torch.zeros_like(differences[:, :1]), differences), dim=1), dim=1
                ).mean()
        elif self.method == "minibatch":
            loss = self._minibatch_loss()
        else:
            loss = self._bigsurv_loss()
        loss = loss + (0.5 * cfg["ridge_lambda"]) * self.beta.square().sum()
        if self.method == "bigsurv_normalized":
            (gradient,) = torch.autograd.grad(loss, self.beta)
            self.optimizer.step(gradient)
        else:
            loss.backward()
            self.optimizer.step()
        self.steps += 1
        if self.method == "bigsurv_normalized":
            with torch.no_grad():
                self.average.add_((self.beta.detach() - self.average) / self.steps)

    def readout(self):
        value = self.average if self.method == "bigsurv_normalized" else self.beta
        return value.detach().numpy().copy()

    def row(self):
        parts = self._parts(self.steps, self.score_rows)
        return dict(
            step=self.steps,
            units_total=sum(parts.values()),
            **{"units_" + name: value for name, value in parts.items()},
        )


def build_engine(method, train, config, seed, cap):
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    parameters = {
        "rho_uniform": {"batch_size", "beta_radius", "decay_steps", "delta", "rho", "shift_bound"},
        "batch_lse": {"outer_batch_size", "risk_batch_size"},
        "minibatch": {"batch_size"},
        "bigsurv_normalized": {"lr_tau", "strata_batch_size", "strata_size"},
        "cox_cc": {"case_batch_size", "n_controls"},
    }[method] | {"learning_rate", "ridge_lambda"}
    missing, unused = parameters - config.keys(), config.keys() - parameters
    if missing or unused:
        raise ValueError(
            f"Invalid {method} configuration: missing {sorted(missing)}, unused {sorted(unused)}"
        )
    if train.X.device.type != "cpu" or train.X.dtype != torch.float64:
        raise ValueError("The benchmark uses float64 CPU data")
    if train.n_events == 0 or cap <= 0:
        raise ValueError("Require at least one event and a positive work cap")
    if config["learning_rate"] <= 0 or config["ridge_lambda"] < 0:
        raise ValueError("Require positive learning rate and nonnegative ridge")
    if method == "rho_uniform":
        return UniformSoftplus(train, config, seed, cap)
    return SampledCox(method, train, config, seed, cap)
