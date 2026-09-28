import os

for name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[name] = "1"

import argparse
from pathlib import Path
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from data import load_dataset
from evaluation import evaluate, validation_metrics, warm_evaluator
from settings import (
    MENU,
    DATASETS,
    RIDGE_LAMBDA,
    TUNING_SEEDS,
    FINAL_SEEDS,
    CHECKPOINT_COUNT,
    CHECKPOINT_START_UNITS,
)
from solvers import METHODS, build_engine


def fit(train, val, method, config, seed, cap, reference):
    started = time.perf_counter()
    engine = build_engine(method, train, dict(config, ridge_lambda=RIDGE_LAMBDA), seed, cap)
    setup_seconds = time.perf_counter() - started
    targets = np.geomspace(CHECKPOINT_START_UNITS, cap, CHECKPOINT_COUNT)
    target_index, optimizer_seconds = 0, 0.0
    rows, betas = [], []

    def record(event):
        rows.append(
            dict(
                engine.row(),
                checkpoint=len(rows),
                event=event,
                wall_time=setup_seconds + optimizer_seconds,
                optimizer_time=optimizer_seconds,
            )
        )
        betas.append(engine.readout())

    record("initial")
    started = time.perf_counter()
    while engine.can_step():
        engine.step()
        if target_index < len(targets) and engine.units >= targets[target_index]:
            optimizer_seconds += time.perf_counter() - started
            while target_index < len(targets) and engine.units >= targets[target_index]:
                target_index += 1
            record("checkpoint")
            started = time.perf_counter()
    optimizer_seconds += time.perf_counter() - started
    if rows[-1]["step"] != engine.steps:
        record("final")
    else:
        rows[-1].update(
            event="final",
            optimizer_time=optimizer_seconds,
            wall_time=setup_seconds + optimizer_seconds,
        )

    started = time.perf_counter()
    for row, beta in zip(rows, betas):
        row.update(evaluate(train, val, beta, reference, RIDGE_LAMBDA))
    curves = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
    if not all(
        np.isfinite(curves[key]).all() for key in ("train_gap", "val_cox_loss", "val_cindex")
    ):
        raise FloatingPointError("Nonfinite full-data diagnostic")
    if curves["train_gap"].min() < -1e-7 or not 0 <= engine.units <= cap:
        raise RuntimeError("Reference gap or work budget check failed")
    selected = int(np.argmin(curves["val_cox_loss"]))
    return dict(
        curves,
        selected_checkpoint=selected,
        selected_beta=betas[selected],
        setup_seconds=setup_seconds,
        evaluation_seconds=time.perf_counter() - started,
    )


def select_configuration(losses):
    means = np.mean(losses, axis=1)
    means[~np.isfinite(means)] = np.inf
    index = int(np.argmin(means))
    if not np.isfinite(means[index]):
        raise RuntimeError("No configuration completed every tuning seed")
    return index, float(means[index])


def experiment(names, output, raw_dir=Path("data"), download=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    counts = {len(MENU[method]) for method in METHODS}
    if len(counts) != 1 or 0 in counts:
        raise ValueError("Methods must have equally sized, nonempty tuning menus")
    count = counts.pop()
    choices, selected_scores = {}, {}

    for name in names:
        case = DATASETS[name]
        train, val, _ = load_dataset(name, raw_dir, download)
        losses = {method: np.full((count, len(TUNING_SEEDS)), np.inf) for method in METHODS}
        for method in METHODS:
            (output / name / method).mkdir(parents=True)
        for index in range(count):
            for column, seed in enumerate(TUNING_SEEDS):
                for offset in range(len(METHODS)):
                    method = METHODS[(offset + index + seed) % len(METHODS)]
                    path = output / name / method / f"tuning_{index}_{seed}.npz"
                    print(f"Tuning {name}: {method}, config {index}, seed {seed}", flush=True)
                    try:
                        result = fit(
                            train,
                            val,
                            method,
                            MENU[method][index],
                            seed,
                            case["tuning_unit_cap"],
                            case["reference"],
                        )
                    except FloatingPointError as error:
                        np.savez(path, status="failed", error=str(error))
                        continue
                    np.savez(path, status="completed", **result)
                    losses[method][index, column] = result["val_cox_loss"][
                        result["selected_checkpoint"]
                    ]
        for method in METHODS:
            key = f"{name}__{method}"
            choices[key], selected_scores[key] = select_configuration(losses[method])
        del train, val, _

    np.savez(output / "choices.npz", **choices)
    np.savez(output / "tuning_scores.npz", **selected_scores)
    models = {}
    for name in names:
        case = DATASETS[name]
        train, val, _ = load_dataset(name, raw_dir, download)
        for seed in FINAL_SEEDS:
            for offset in range(len(METHODS)):
                method = METHODS[(offset + seed) % len(METHODS)]
                index = choices[f"{name}__{method}"]
                print(f"Final {name}: {method}, config {index}, seed {seed}", flush=True)
                path = output / name / method / f"final_{seed}.npz"
                try:
                    result = fit(
                        train,
                        val,
                        method,
                        MENU[method][index],
                        seed,
                        case["unit_cap"],
                        case["reference"],
                    )
                except FloatingPointError as error:
                    np.savez(path, status="failed", error=str(error))
                    raise
                np.savez(path, status="completed", config_index=index, **result)
                models[f"{name}__{method}__{seed}"] = result["selected_beta"]
        del train, val, _


    np.savez(output / "models.npz", **models)
    for name in names:
        train, val, test = load_dataset(name, raw_dir, download)
        del train, val
        for method in METHODS:
            for seed in FINAL_SEEDS:
                beta = models[f"{name}__{method}__{seed}"]
                metrics = validation_metrics(test, beta)
                np.savez(
                    output / name / method / f"test_{seed}.npz",
                    test_cox_loss=metrics["val_cox_loss"],
                    test_cindex=metrics["val_cindex"],
                )
        del test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("all", *DATASETS), default="all")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--download", action="store_true", help="Download missing real-data files")
    args = parser.parse_args()
    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        warm_evaluator()
        experiment(names, args.output, args.data_dir, args.download)


if __name__ == "__main__":
    main()
