from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from sklearn.model_selection import train_test_split

from settings import DATASETS


@dataclass
class CoxData:
    X: Tensor
    durations: Tensor
    events: Tensor
    X_sorted: Tensor
    durations_sorted: Tensor
    events_sorted: Tensor
    event_indices: Tensor
    risk_end: Tensor
    n: int
    d: int
    n_events: int

    @classmethod
    def from_arrays(cls, X, durations, events, device="cpu", dtype=torch.float64):
        X = torch.as_tensor(X, dtype=dtype, device=device).contiguous()
        durations = torch.as_tensor(durations, dtype=torch.float64, device=device).contiguous()
        raw_events = torch.as_tensor(events, device=device)
        if (
            X.ndim != 2
            or durations.ndim != 1
            or raw_events.shape != durations.shape
            or X.shape[0] != durations.numel()
        ):
            raise ValueError("X, durations, events have incompatible shapes")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("Empty data or no nonconstant covariates")
        if not bool(torch.isfinite(X).all() & torch.isfinite(durations).all()) or bool(
            (durations < 0).any()
        ):
            raise ValueError("Covariates/times must be finite and times nonnegative")
        if not bool(((raw_events == 0) | (raw_events == 1)).all()):
            raise ValueError("events must be binary")
        events = raw_events.to(torch.bool).contiguous()
        order = torch.argsort(durations, descending=True, stable=True)
        ts = durations[order].contiguous()
        es = events[order].contiguous()
        ends = torch.searchsorted(-ts, -ts, right=True)
        return cls(
            X,
            durations,
            events,
            X[order].contiguous(),
            ts,
            es,
            torch.nonzero(es, as_tuple=True)[0],
            ends,
            X.shape[0],
            X.shape[1],
            int(es.sum()),
        )


class Preprocessor:

    def fit(self, X):
        frame = pd.DataFrame(X).copy()
        frame.columns = frame.columns.map(str)
        self.columns = list(frame.columns)
        self.numeric = list(frame.select_dtypes(include=[np.number, "bool"]).columns)
        self.categories = {
            c: sorted(frame[c].dropna().astype(str).unique().tolist())
            for c in self.columns
            if c not in self.numeric
        }
        numeric = frame[self.numeric].apply(pd.to_numeric).replace([np.inf, -np.inf], np.nan)
        self.medians = numeric.median().fillna(0).to_dict()
        encoded = self._encode(frame)
        self.mean = encoded.mean(axis=0)
        self.scale = encoded.std(axis=0)
        self.keep = self.scale > 1e-12
        if not self.keep.any():
            raise ValueError("All covariates are constant in training split")
        return self

    def _encode(self, frame):
        frame = pd.DataFrame(frame).copy()
        frame.columns = frame.columns.map(str)
        blocks = []
        if self.numeric:
            numeric = frame[self.numeric].apply(pd.to_numeric).replace([np.inf, -np.inf], np.nan)
            blocks.append(numeric.fillna(self.medians).to_numpy(dtype=np.float64))
        for c, levels in self.categories.items():
            values = frame[c].fillna("<MISSING>").astype(str).to_numpy()

            blocks.append(values[:, None] == np.asarray(levels + ["<MISSING>"])[None, :])
        return np.concatenate(blocks, axis=1).astype(np.float64)

    def transform(self, X):
        z = self._encode(X)
        return np.ascontiguousarray(
            (z[:, self.keep] - self.mean[self.keep]) / self.scale[self.keep]
        )


def prepare_splits(X, durations, events, seed=1701, val_fraction=0.2, test_fraction=0.2):
    frame = pd.DataFrame(X)
    durations, events = np.asarray(durations), np.asarray(events)
    ids = np.arange(len(frame))
    train, held = train_test_split(
        ids, test_size=val_fraction + test_fraction, random_state=seed, stratify=events
    )
    val, test = train_test_split(
        held,
        test_size=test_fraction / (test_fraction + val_fraction),
        random_state=seed + 1,
        stratify=events[held],
    )
    pre = Preprocessor().fit(frame.iloc[train])
    return tuple(
        CoxData.from_arrays(pre.transform(frame.iloc[idx]), durations[idx], events[idx])
        for idx in (train, val, test)
    )


def read_real(path, name, spec):
    frame = pd.read_csv(path)
    for column in spec["categories"]:
        frame[column] = frame[column].map(lambda x: str(x) if pd.notna(x) else np.nan)
    if name == "support2":
        frame["race"] = frame.race.fillna("missing")
        frame = frame.dropna(subset=spec["features"] + [spec["duration"], spec["event"]])
    else:
        frame = frame.dropna(subset=[spec["duration"], spec["event"]])
    X = frame[spec["features"]]
    durations = frame[spec["duration"]].to_numpy(dtype=float)
    events = pd.to_numeric(frame[spec["event"]]).to_numpy() == 1
    return X, durations, events


def input_fingerprint(X, durations, events):
    payload = dict(
        columns=list(X.columns),
        values=X.to_numpy().tolist(),
        durations=durations.tolist(),
        events=events.tolist(),
    )
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
    ).hexdigest()


def synthetic_cox(spec):
    n, d = spec["n"], spec["d"]
    streams = np.random.SeedSequence(spec["seed"]).spawn(3)
    rng, pilot, coeff = (np.random.default_rng(s) for s in streams)
    beta = coeff.normal(size=d)
    beta *= spec["signal"] / np.linalg.norm(beta)
    X = rng.normal(size=(n, d))
    eta = X @ beta
    shape = spec["baseline_shape"]
    failure = np.power(rng.exponential(size=n) * np.exp(-eta), 1.0 / shape)
    pilot_eta = pilot.normal(scale=spec["signal"], size=65536)
    pilot_time = np.power(pilot.exponential(size=65536) * np.exp(-pilot_eta), 1.0 / shape)
    low, high = -40.0, 40.0
    for _ in range(80):
        middle = (low + high) / 2
        fraction = np.mean(-np.expm1(-np.exp(middle) * pilot_time))
        if fraction < spec["censoring"]:
            low = middle
        else:
            high = middle
    rate = float(np.exp((low + high) / 2))
    censor = rng.exponential(scale=1 / rate, size=n)
    durations = np.minimum(failure, censor)
    events = (failure <= censor).astype(np.int64)
    correlation = spec["feature_correlation"]
    if correlation:
        covariance = correlation ** np.abs(np.arange(d)[:, None] - np.arange(d)[None, :])
        X = X @ np.linalg.cholesky(covariance).T
    return np.ascontiguousarray(X), durations, events


def download_csv(spec, target):
    with urllib.request.urlopen(spec["url"], timeout=60) as response:
        content = response.read()
    if spec["url"].endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            matches = [name for name in archive.namelist() if Path(name).name == spec["file"]]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one {spec['file']} in archive")
            content = archive.read(matches[0])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)


def load_dataset(name, raw_dir=Path("data"), download=False):
    case = DATASETS[name]
    spec = case["data"]
    if spec["kind"] == "real":
        path = Path(raw_dir) / spec["file"]
        if not path.exists():
            if not download:
                raise FileNotFoundError(f"Place {spec['file']} in {raw_dir}, or pass --download")
            download_csv(spec, path)
        X, durations, events = read_real(path, name, spec)
        if input_fingerprint(X, durations, events) != case["input_sha256"]:
            raise ValueError(f"Raw values or row order differ from the article: {path}")
    else:
        X, durations, events = synthetic_cox(spec)
    train, val, test = prepare_splits(X, durations, events)
    for split, data in (("train", train), ("val", val), ("test", test)):
        if data.n != case["n_" + split] or data.d != case["d"]:
            raise ValueError(f"Unexpected shape for {name}/{split}")
    if train.n_events != case["n_events"]:
        raise ValueError(f"Training event count differs from the article: {name}")
    return train, val, test
