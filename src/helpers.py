import time
import tracemalloc

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow import keras

from src.data import load_california


class SGD:
    def __init__(
            self,
            learning_rate=0.001,
            epochs=20,
            batch_size=32,
            regularization=None,
            reg_param=0.01,
            lr_schedule=None,
            random_state=None,
    ):
        self.flops_count = 0
        self.memory_used = 0
        self.training_time = 0.0

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.regularization = regularization
        self.reg_param = reg_param
        self.lr_schedule = lr_schedule
        self.rng = np.random.RandomState(random_state)

        self.weights = None
        self.bias = None

        self.loss_history = []

        self._lr_decay_rates = {"time_decay": 0.01, "exponential": 0.01}
        self._step_decay = {"drop": 0.5, "epochs_drop": 10}

    def _initialize(self, n_features):
        limit = np.sqrt(6.0 / (n_features + 1))
        self.weights = self.rng.uniform(-limit, limit, n_features)
        self.bias = 0.0

    def _get_lr(self, epoch):
        if not self.lr_schedule or self.lr_schedule == "constant":
            return self.learning_rate

        schedule = {
            "time_decay": lambda e: self.learning_rate / (1 + self._lr_decay_rates["time_decay"] * e),
            "step_decay": lambda e: self.learning_rate
                                    * np.power(self._step_decay["drop"],
                                               np.floor(e / self._step_decay["epochs_drop"])),
            "exponential": lambda e: self.learning_rate * np.exp(-self._lr_decay_rates["exponential"] * e),
        }

        return schedule.get(self.lr_schedule, lambda e: self.learning_rate)(epoch)

    def _apply_reg(self, w, grad):
        if self.regularization is None:
            return grad
        if self.regularization == "l2":
            return grad + self.reg_param * w
        if self.regularization == "l1":
            return grad + self.reg_param * np.sign(w)
        l1r = 0.5
        return grad + self.reg_param * (l1r * np.sign(w) + (1 - l1r) * w)

    def fit(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        m, n = x.shape

        self._initialize(n)

        tracemalloc.start()
        t0 = time.perf_counter()

        self.loss_history = []

        for epoch in range(self.epochs):
            idx = self.rng.permutation(m)
            xs, ys = x[idx], y[idx]
            lr = self._get_lr(epoch)

            for start in range(0, m, self.batch_size):
                xb = xs[start: start + self.batch_size]
                yb = ys[start: start + self.batch_size]

                pred = xb @ self.weights + self.bias
                err = pred - yb

                dw = (xb.T @ err) / xb.shape[0]
                norm_dw = np.linalg.norm(dw)
                if norm_dw > 1.0:
                    dw = dw * (1.0 / norm_dw)
                dw = self._apply_reg(self.weights, dw)
                db = np.mean(err)

                self.weights -= lr * dw
                self.bias -= lr * db

            full_pred = x @ self.weights + self.bias
            loss = np.mean((full_pred - y) ** 2)
            self.loss_history.append(loss)

        t1 = time.perf_counter()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        self.training_time = t1 - t0
        self.memory_used = peak

        ops_per_sample = 2 * n + (2 * n + 1) + 2
        self.flops_count = self.epochs * m * ops_per_sample

        return self

    def predict(self, X):
        return np.asarray(X) @ self.weights + self.bias


def test_model_performance(models):
    x, y = load_california(scale=True)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    results = []
    for i, m in enumerate(models):
        m.fit(x_train, y_train)

        y_pred = m.predict(x_test)
        if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
            y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e10, neginf=-1e10)

        mse = mean_squared_error(y_test, y_pred)
        results.append({
            "model_idx": i,
            "batch_size": m.batch_size,
            "mse": mse,
            "time_sec": m.training_time,
            "mem_bytes": m.memory_used,
            "flops": m.flops_count,
        })

    return pd.DataFrame(results).sort_values("batch_size")


def plot_performance_metrics(models, save_path=None):
    reg_info = f"reg: {models[1].regularization}" if models[1].regularization else ""
    lr_info = f"lrs: {models[1].lr_schedule}" if models[1].lr_schedule else ""
    print(f"Parameters - {reg_info}, {lr_info}")

    df = test_model_performance(models)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    specs = [
        ("mse", "MSE", False),
        ("time_sec", "Time (s)", False),
        ("mem_bytes", "Memory (bytes)", True),
        ("flops", "FLOPs", True),
    ]

    for ax, (col, label, log_scale) in zip(axes.flatten(), specs):
        ax.scatter(df["batch_size"], df[col], s=50, alpha=0.7)
        ax.plot(df["batch_size"], df[col], alpha=0.7)
        ax.set_xscale("log", base=2)
        if log_scale:
            ax.set_yscale("log")
        ax.set_xlabel("batch_size")
        ax.set_ylabel(label)
        ax.set_title(f"{label} vs batch_size")
        ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


# TensorFlow and PyTorch training functions

def init_torch_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        init.zeros_(m.bias)


def make_torch_optim(name, params, lr_dict):
    lr = lr_dict[name]
    if name.startswith("SGD"):
        momentum = 0.9 if ("Momentum" in name or "Nesterov" in name) else 0.0
        nesterov = "Nesterov" in name
        return optim.SGD(params, lr=lr, momentum=momentum, nesterov=nesterov)
    return {
        "Adagrad": optim.Adagrad,
        "RMSprop": optim.RMSprop,
        "Adam": optim.Adam,
    }[name](params, lr=lr)


class LinReg(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.lin = nn.Linear(d_in, 1)
        self.apply(init_torch_weights)

    def forward(self, x):
        return self.lin(x)


def train_torch(opt_name, xtr_t, ytr_t, xte_t, yte_t, epochs, batch, lr_dict):
    model = LinReg(xtr_t.shape[1])
    loss_fn = nn.MSELoss()
    opt = make_torch_optim(opt_name, model.parameters(), lr_dict)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(xtr_t, ytr_t),
        batch_size=batch,
        shuffle=True
    )

    baseline_mem = psutil.Process().memory_info().rss
    peak_mem = baseline_mem
    t0 = time.time()
    train_losses, test_losses = [], []

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            train_losses.append(loss_fn(model(xtr_t), ytr_t).item())
            test_losses.append(loss_fn(model(xte_t), yte_t).item())
        peak_mem = max(peak_mem, psutil.Process().memory_info().rss)

    return {
        "optimizer": opt_name,
        "framework": "PyTorch",
        "final_mse": test_losses[-1],
        "train_losses": train_losses,
        "test_losses": test_losses,
        "training_time": time.time() - t0,
        "memory_used": (peak_mem - baseline_mem) / 1024 ** 2,
    }


def make_tf_optim(name, lr_dict):
    lr = lr_dict[name]
    if name.startswith("SGD"):
        use_momentum = "Momentum" in name or "Nesterov" in name
        use_nesterov = "Nesterov" in name
        opt = keras.optimizers.SGD(learning_rate=lr)
        if hasattr(opt, 'nesterov'):
            opt.nesterov = use_nesterov
        if use_momentum and hasattr(opt, 'momentum'):
            opt.momentum = 0.9

        return opt

    return {
        "Adagrad": keras.optimizers.Adagrad,
        "RMSprop": keras.optimizers.RMSprop,
        "Adam": keras.optimizers.Adam,
    }[name](learning_rate=lr)


def create_tf_model(d_in):
    return keras.Sequential([
        keras.layers.Input(shape=(d_in,)),
        keras.layers.Dense(
            1,
            kernel_initializer=keras.initializers.GlorotUniform(seed=42),
            bias_initializer=keras.initializers.Zeros(),
        )
    ])


def train_tf(opt_name, xtr_tf, ytr_tf, xte_tf, yte_tf, epochs, batch, lr_dict):
    model = create_tf_model(xtr_tf.shape[1])
    opt = make_tf_optim(opt_name, lr_dict)
    loss_fn = keras.losses.MeanSquaredError()

    baseline_mem = psutil.Process().memory_info().rss
    peak_mem = baseline_mem
    t0 = time.time()

    ds = tf.data.Dataset.from_tensor_slices((xtr_tf, ytr_tf))
    ds = ds.shuffle(len(xtr_tf)).batch(batch)

    train_losses, test_losses = [], []
    for _ in range(epochs):
        for xb, yb in ds:
            with tf.GradientTape() as tape:
                pred = model(xb, training=True)
                loss = loss_fn(yb, pred)
            grads = tape.gradient(loss, model.trainable_variables)
            grads = [tf.clip_by_norm(g, 1.0) for g in grads]
            opt.apply_gradients(zip(grads, model.trainable_variables))

        tr_pred = model(xtr_tf, training=False)
        te_pred = model(xte_tf, training=False)
        train_losses.append(loss_fn(ytr_tf, tr_pred).numpy())
        test_losses.append(loss_fn(yte_tf, te_pred).numpy())
        peak_mem = max(peak_mem, psutil.Process().memory_info().rss)

    return {
        "optimizer": opt_name,
        "framework": "TensorFlow",
        "final_mse": test_losses[-1],
        "train_losses": train_losses,
        "test_losses": test_losses,
        "training_time": time.time() - t0,
        "memory_used": (peak_mem - baseline_mem) / 1024 ** 2,
    }


def summarise(results):
    print(f"{'Opt':<15}{'Framework':<12}{'MSE':<14}{'Time(s)':<9}{'Mem(MB)':<8}")
    print("-" * 59)
    for r in results:
        print(
            f"{r['optimizer']:<15}{r['framework']:<12}"
            f"{r['final_mse']:<14.4e}"
            f"{r['training_time']:<9.2f}"
            f"{r['memory_used']:<8.2f}"
        )


def plot(results):
    torch_r = [r for r in results if r["framework"] == "PyTorch"]
    tf_r = [r for r in results if r["framework"] == "TensorFlow"]
    names = [r["optimizer"] for r in torch_r]
    x = np.arange(len(torch_r))
    w = 0.35

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0, 0].bar(x - w / 2, [r["final_mse"] for r in torch_r], w, label="PyTorch")
    axs[0, 0].bar(x + w / 2, [r["final_mse"] for r in tf_r], w, label="TensorFlow")
    axs[0, 0].set_yscale("log")
    axs[0, 0].set_title("Final Test MSE (log)")
    axs[0, 0].set_xticks(x, names, rotation=45)
    axs[0, 0].legend()
    axs[0, 0].grid(True, which="both")

    axs[0, 1].bar(x - w / 2, [r["training_time"] for r in torch_r], w)
    axs[0, 1].bar(x + w / 2, [r["training_time"] for r in tf_r], w)
    axs[0, 1].set_title("Training Time (s)")
    axs[0, 1].set_xticks(x, names, rotation=45)
    axs[0, 1].grid(True)

    axs[1, 0].bar(x - w / 2, [r["memory_used"] for r in torch_r], w)
    axs[1, 0].bar(x + w / 2, [r["memory_used"] for r in tf_r], w)
    axs[1, 0].set_title("Extra Memory (MB)")
    axs[1, 0].set_xticks(x, names, rotation=45)
    axs[1, 0].grid(True)

    for r in torch_r:
        axs[1, 1].plot(r["test_losses"], label=r["optimizer"])
    axs[1, 1].set_title("PyTorch Convergence")
    axs[1, 1].set_xlabel("Epoch")
    axs[1, 1].set_ylabel("MSE")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.show()
