import time
import psutil
import numpy as np
import pandas as pd
import optuna
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from src.data import load_california


def model_objective(trial, model):
    x, y = load_california(scale=True)

    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    batch_size = model.batch_size

    proc = psutil.Process()
    mem_before = proc.memory_info().rss
    t0 = time.perf_counter()

    y_pred = model.predict(X_test)

    t1 = time.perf_counter()
    mem_after = proc.memory_info().rss

    mse = mean_squared_error(y_test, y_pred)

    n_batches = int(np.ceil(len(X_train) / batch_size))
    flops_one_batch = 2 * X_train.shape[1] * batch_size
    total_flops = flops_one_batch * n_batches * model.epochs

    trial.set_user_attr("batch_size", batch_size)
    trial.set_user_attr("time_sec", t1 - t0)
    trial.set_user_attr("mem_bytes", mem_after - mem_before)
    trial.set_user_attr("flops", total_flops)

    return mse


def test_model_performance(models):
    study = optuna.create_study(direction="minimize")

    for _ in models:
        study.enqueue_trial({})

    study.optimize(lambda trial: model_objective(trial, models[trial.number]),
                   n_trials=len(models),
                   show_progress_bar=True)

    rows = []
    for t in study.trials:
        rows.append({
            "batch_size": t.user_attrs["batch_size"],
            "mse": t.value,
            "time_sec": t.user_attrs["time_sec"],
            "mem_bytes": t.user_attrs["mem_bytes"],
            "flops": t.user_attrs["flops"]
        })

    return pd.DataFrame(rows).sort_values("batch_size")


def plot_performance_metrics(models):
    df = test_model_performance(models)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = [("mse", "MSE"),
               ("time_sec", "Time (s)"),
               ("mem_bytes", "Δ Memory (bytes)"),
               ("flops", "Estimated FLOPs")]

    for ax, (col, label) in zip(axes.flatten(), metrics):
        ax.plot(df["batch_size"], df[col], "-o")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("batch_size")
        ax.set_ylabel(label)
        ax.set_title(f"{label} vs batch_size")
        ax.grid(True)

    plt.tight_layout()
    plt.show()
