import time
import psutil
import numpy as np
import pandas as pd
import optuna
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from src.data import load_california
import time, psutil, numpy as np, matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.optim as optim, torch.nn.init as init


class SGD:
    def __init__(self, learning_rate=0.01, epochs=100, batch_size=32, regularization=None, reg_param=0.01,
                 lr_schedule=None, random_state=None, early_stopping=False, patience=5):
        """
        Инициализация SGD с настройками

        Параметры:
        learning_rate : float - начальная скорость обучения
        epochs : int - количество эпох обучения
        batch_size : int - размер мини-батча
        regularization : str или None - тип регуляризации ('l1', 'l2', 'elasticnet')
        reg_param : float - параметр регуляризации (lambda)
        lr_schedule : str или None - график скорости обучения ('constant', 'time_decay', 'step_decay', 'exponential')
        random_state : int или None - инициализация генератора случайных чисел
        early_stopping : bool - включение ранней остановки обучения
        patience : int - количество эпох без улучшения для ранней остановки
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.regularization = regularization
        self.reg_param = reg_param
        self.lr_schedule = lr_schedule
        self.early_stopping = early_stopping
        self.patience = patience

        self.rng = np.random.RandomState(random_state)

        self.weights = None
        self.bias = None

        self.loss_history = []
        self.val_loss_history = []
        self.best_weights = None
        self.best_bias = None
        self.best_val_loss = float('inf')
        self.best_epoch = 0

        self._lr_decay_rates = {
            'time_decay': 0.01,
            'exponential': 0.01
        }
        self._step_decay_params = {
            'drop': 0.5,
            'epochs_drop': 10
        }

        self._elasticnet_l1_ratio = 0.5

    def _initialize_parameters(self, n_features):
        limit = np.sqrt(6 / (n_features + 1))
        self.weights = self.rng.uniform(-limit, limit, n_features)
        self.bias = 0.0

        self.best_weights = self.weights.copy()
        self.best_bias = self.bias

    def _get_learning_rate(self, epoch):
        if self.lr_schedule is None or self.lr_schedule == 'constant':
            return self.learning_rate

        lr_schedulers = {
            'time_decay': lambda e: self.learning_rate / (1 + self._lr_decay_rates['time_decay'] * e),
            'step_decay': lambda e: self.learning_rate * np.power(
                self._step_decay_params['drop'],
                np.floor(e / self._step_decay_params['epochs_drop'])
            ),
            'exponential': lambda e: self.learning_rate * np.exp(-self._lr_decay_rates['exponential'] * e)
        }

        return lr_schedulers.get(self.lr_schedule, lambda e: self.learning_rate)(epoch)

    def _apply_regularization(self, weights, gradient):
        if self.regularization is None:
            return gradient

        regularizers = {
            'l2': lambda w, g: g + self.reg_param * w,
            'l1': lambda w, g: g + self.reg_param * np.sign(w),
            'elasticnet': lambda w, g: g + self.reg_param * (
                    self._elasticnet_l1_ratio * np.sign(w) +
                    (1 - self._elasticnet_l1_ratio) * w
            )
        }

        return regularizers.get(self.regularization, lambda w, g: g)(weights, gradient)

    def _compute_gradient(self, x_batch, y_batch):
        m = x_batch.shape[0]
        predictions = x_batch @ self.weights + self.bias
        errors = predictions - y_batch

        dw = (x_batch.T @ errors) / m
        db = np.mean(errors)

        dw = self._apply_regularization(self.weights, dw)

        return dw, db

    def fit(self, x, y, x_val=None, y_val=None, verbose=0):
        x = np.asarray(x)
        y = np.asarray(y)

        use_validation = False
        if x_val is not None and y_val is not None:
            x_val = np.asarray(x_val)
            y_val = np.asarray(y_val)
            use_validation = True

        n_samples, n_features = x.shape
        self._initialize_parameters(n_features)

        self.loss_history = []
        self.val_loss_history = []

        no_improvement_count = 0
        for epoch in range(self.epochs):
            indices = self.rng.permutation(n_samples)
            x_shuffled = x[indices]
            y_shuffled = y[indices]

            current_lr = self._get_learning_rate(epoch)

            for i in range(0, n_samples, self.batch_size):
                end_idx = min(i + self.batch_size, n_samples)
                x_batch = x_shuffled[i:end_idx]
                y_batch = y_shuffled[i:end_idx]

                dw, db = self._compute_gradient(x_batch, y_batch)
                self.weights -= current_lr * dw
                self.bias -= current_lr * db

            train_predictions = x @ self.weights + self.bias
            train_loss = np.mean((train_predictions - y) ** 2)
            self.loss_history.append(train_loss)

            val_loss = None
            if use_validation:
                val_predictions = x_val @ self.weights + self.bias
                val_loss = np.mean((val_predictions - y_val) ** 2)
                self.val_loss_history.append(val_loss)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_weights = self.weights.copy()
                    self.best_bias = self.bias
                    self.best_epoch = epoch
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

            if verbose > 0 and epoch % verbose == 0:
                val_str = f", val_loss: {val_loss:.6f}" if val_loss is not None else ""
                print(f"Epoch {epoch}/{self.epochs} - loss: {train_loss:.6f}{val_str}")

            if self.early_stopping and use_validation and no_improvement_count >= self.patience:
                if verbose > 0:
                    print(f"Early stopping on epoch {epoch}")
                break

        if use_validation:
            self.weights = self.best_weights
            self.bias = self.best_bias
            if verbose > 0:
                print(f"Best model on epoch {self.best_epoch} with val_loss: {self.best_val_loss:.6f}")

        return self

    def predict(self, x):
        x = np.asarray(x)
        return x @ self.weights + self.bias


def model_objective(trial, model):
    x, y = load_california(scale=True)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    batch_size = model.batch_size

    proc = psutil.Process()
    mem_before = proc.memory_info().rss
    t0 = time.perf_counter()

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    t1 = time.perf_counter()
    mem_after = proc.memory_info().rss
    training_time = t1 - t0
    memory_used = mem_after - mem_before

    mse = mean_squared_error(y_test, y_pred)

    n_samples, n_features = x_train.shape
    n_batches_per_epoch = int(np.ceil(n_samples / batch_size))

    forward_ops = n_features + (n_features - 1)
    gradient_ops = 2 * n_features
    weight_update_ops = 2 * (n_features + 1)

    ops_per_batch = batch_size * (forward_ops + gradient_ops) + weight_update_ops
    total_flops = ops_per_batch * n_batches_per_epoch * model.epochs

    trial.set_user_attr("batch_size", batch_size)
    trial.set_user_attr("time_sec", training_time)
    trial.set_user_attr("mem_bytes", memory_used)
    trial.set_user_attr("flops", total_flops)

    return mse


def test_model_performance(models, direction="minimize", n_jobs=-1, cache_results=False,
                           cache_path="model_performance_results.csv"):
    if models:
        model = models[0]
        study_name = f"SGD_reg-{model.regularization or 'none'}_lr-{model.learning_rate}"

        if model.lr_schedule:
            study_name += f"_schedule-{model.lr_schedule}"

        if model.regularization:
            study_name += f"_lambda-{model.reg_param}"

        if model.early_stopping:
            study_name += f"_early-stop-{model.patience}"
    else:
        study_name = "SGD_performance_study"

    study = optuna.create_study(direction=direction, study_name=study_name)

    for _ in models:
        study.enqueue_trial({})

    study.optimize(
        lambda trial: model_objective(trial, models[trial.number]),
        n_trials=len(models),
        n_jobs=n_jobs,
        show_progress_bar=True
    )

    rows = []
    for i, t in enumerate(study.trials):
        rows.append({
            "model_idx": i,
            "batch_size": t.user_attrs["batch_size"],
            "mse": t.value,
            "time_sec": t.user_attrs["time_sec"],
            "mem_bytes": t.user_attrs["mem_bytes"],
            "flops": t.user_attrs["flops"],
            "time_per_epoch": t.user_attrs["time_sec"] / models[i].epochs if models[i].epochs > 0 else 0,
            "study_name": study.study_name  # Сохраняем название исследования в результатах
        })

    results = pd.DataFrame(rows).sort_values("batch_size")

    if cache_results and cache_path:
        results.to_csv(cache_path, index=False)

    return results


def plot_performance_metrics(models, save_path=None):
    df = test_model_performance(models)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = [("mse", "MSE"),
               ("time_sec", "Time (s)"),
               ("mem_bytes", "Δ Memory (bytes)"),
               ("flops", "Estimated FLOPs")]

    for ax, (col, label) in zip(axes.flatten(), metrics):
        ax.scatter(df["batch_size"], df[col], s=60, alpha=0.7)
        ax.plot(df["batch_size"], df[col], "-", alpha=0.7)
        ax.set_xscale("log", base=2)

        if col in ["flops", "mem_bytes"]:
            ax.set_yscale("log")

        ax.set_xlabel("batch_size")
        ax.set_ylabel(label)
        ax.set_title(f"{label} vs batch_size")
        ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()

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
        "Adam":    optim.Adam,
    }[name](params, lr=lr)

class LinReg(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.lin = nn.Linear(d_in, 1)
        self.apply(init_torch_weights)

    def forward(self, x):
        return self.lin(x)

def train_torch(opt_name, Xtr_t, ytr_t, Xte_t, yte_t, epochs, batch, lr_dict):
    model = LinReg(Xtr_t.shape[1])
    loss_fn = nn.MSELoss()
    opt = make_torch_optim(opt_name, model.parameters(), lr_dict)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xtr_t, ytr_t),
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
            train_losses.append(loss_fn(model(Xtr_t), ytr_t).item())
            test_losses.append(loss_fn(model(Xte_t), yte_t).item())
        peak_mem = max(peak_mem, psutil.Process().memory_info().rss)

    return {
        "optimizer": opt_name,
        "framework": "PyTorch",
        "final_mse": test_losses[-1],
        "train_losses": train_losses,
        "test_losses": test_losses,
        "training_time": time.time() - t0,
        "memory_used": (peak_mem - baseline_mem) / 1024**2,
    }

def make_tf_optim(name, lr_dict):
    lr = lr_dict[name]
    if name.startswith("SGD"):
        kwargs = {"nesterov": "Nesterov" in name}
        if "Momentum" in name or "Nesterov" in name:
            kwargs["momentum"] = 0.9
        return keras.optimizers.SGD(learning_rate=lr, **kwargs)
    return {
        "Adagrad": keras.optimizers.Adagrad,
        "RMSprop": keras.optimizers.RMSprop,
        "Adam":    keras.optimizers.Adam,
    }[name](learning_rate=lr)

def create_tf_model(d_in):
    return keras.Sequential([
        keras.layers.Dense(
            1,
            input_shape=(d_in,),
            kernel_initializer=keras.initializers.GlorotUniform(seed=42),
            bias_initializer=keras.initializers.Zeros(),
        )
    ])

def train_tf(opt_name, Xtr_tf, ytr_tf, Xte_tf, yte_tf, epochs, batch, lr_dict):
    model = create_tf_model(Xtr_tf.shape[1])
    opt = make_tf_optim(opt_name, lr_dict)
    loss_fn = keras.losses.MeanSquaredError()

    baseline_mem = psutil.Process().memory_info().rss
    peak_mem = baseline_mem
    t0 = time.time()

    ds = tf.data.Dataset.from_tensor_slices((Xtr_tf, ytr_tf))
    ds = ds.shuffle(len(Xtr_tf)).batch(batch)

    train_losses, test_losses = [], []
    for _ in range(epochs):
        for xb, yb in ds:
            with tf.GradientTape() as tape:
                pred = model(xb, training=True)
                loss = loss_fn(yb, pred)
            grads = tape.gradient(loss, model.trainable_variables)
            grads = [tf.clip_by_norm(g, 1.0) for g in grads]
            opt.apply_gradients(zip(grads, model.trainable_variables))

        tr_pred = model(Xtr_tf, training=False)
        te_pred = model(Xte_tf, training=False)
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
        "memory_used": (peak_mem - baseline_mem) / 1024**2,
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

    axs[0, 0].bar(x - w/2, [r["final_mse"] for r in torch_r], w, label="PyTorch")
    axs[0, 0].bar(x + w/2, [r["final_mse"] for r in tf_r], w, label="TensorFlow")
    axs[0, 0].set_yscale("log")
    axs[0, 0].set_title("Final Test MSE (log)")
    axs[0, 0].set_xticks(x, names, rotation=45)
    axs[0, 0].legend(); axs[0, 0].grid(True, which="both")

    axs[0, 1].bar(x - w/2, [r["training_time"] for r in torch_r], w)
    axs[0, 1].bar(x + w/2, [r["training_time"] for r in tf_r], w)
    axs[0, 1].set_title("Training Time (s)")
    axs[0, 1].set_xticks(x, names, rotation=45); axs[0, 1].grid(True)

    axs[1, 0].bar(x - w/2, [r["memory_used"] for r in torch_r], w)
    axs[1, 0].bar(x + w/2, [r["memory_used"] for r in tf_r], w)
    axs[1, 0].set_title("Extra Memory (MB)")
    axs[1, 0].set_xticks(x, names, rotation=45); axs[1, 0].grid(True)

    for r in torch_r:
        axs[1, 1].plot(r["test_losses"], label=r["optimizer"])
    axs[1, 1].set_title("PyTorch Convergence")
    axs[1, 1].set_xlabel("Epoch"); axs[1, 1].set_ylabel("MSE")
    axs[1, 1].legend(); axs[1, 1].grid(True)

    plt.tight_layout()
    plt.show()