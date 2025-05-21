import time
import psutil
import numpy as np
import pandas as pd
import optuna
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from src.data import load_california

class SGD:
    """Стохастический градиентный спуск для задач регрессии."""

    def __init__(self, learning_rate=0.01, epochs=100, batch_size=32,
                 regularization=None, reg_param=0.01, lr_schedule=None,
                 momentum=0.0, early_stopping=False, patience=10,
                 verbose=False, random_state=None):
        """
        Инициализация SGD с настройками

        Параметры:
        learning_rate : начальная скорость обучения
        epochs : количество эпох обучения
        batch_size : размер мини-батча
        regularization : тип регуляризации ('l1', 'l2', 'elasticnet')
        reg_param : параметр регуляризации (lambda)
        lr_schedule : график скорости обучения ('constant', 'time_decay', 'step_decay', 'exponential')
        momentum : коэффициент импульса (0 - без импульса)
        early_stopping : использовать раннюю остановку
        patience : количество эпох без улучшения для ранней остановки
        verbose : выводить информацию о процессе обучения
        random_state : значение для воспроизводимости результатов
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.regularization = regularization
        self.reg_param = reg_param
        self.lr_schedule = lr_schedule
        self.momentum = momentum
        self.early_stopping = early_stopping
        self.patience = patience
        self.verbose = verbose

        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        self.weights = None
        self.bias = None
        self.velocity_w = None
        self.velocity_b = None

        self.loss_history = []
        self.val_loss_history = []
        self.best_weights = None
        self.best_bias = None
        self.best_val_loss = float('inf')
        self.best_epoch = 0

    def _initialize_parameters(self, n_features):
        """Инициализация весов и смещения"""
        # Инициализация Xavier для лучшей сходимости
        limit = np.sqrt(6 / (n_features + 1))
        self.weights = self.rng.uniform(-limit, limit, n_features)
        self.bias = 0.0

        # Инициализация для импульса
        self.velocity_w = np.zeros(n_features)
        self.velocity_b = 0.0

        # Для early stopping
        self.best_weights = self.weights.copy()
        self.best_bias = self.bias

    def _get_learning_rate(self, epoch):
        """Расчет скорости обучения в зависимости от графика"""
        if self.lr_schedule is None or self.lr_schedule == 'constant':
            return self.learning_rate

        if self.lr_schedule == 'time_decay':
            decay_rate = 0.01
            return self.learning_rate / (1 + decay_rate * epoch)

        if self.lr_schedule == 'step_decay':
            drop = 0.5
            epochs_drop = 10
            return self.learning_rate * np.power(drop, np.floor(epoch / epochs_drop))

        if self.lr_schedule == 'exponential':
            decay_rate = 0.01
            return self.learning_rate * np.exp(-decay_rate * epoch)

        return self.learning_rate

    def _apply_regularization(self, weights, gradient):
        """Применение регуляризации к градиенту"""
        if self.regularization is None:
            return gradient

        if self.regularization == 'l2':
            return gradient + self.reg_param * weights

        if self.regularization == 'l1':
            return gradient + self.reg_param * np.sign(weights)

        if self.regularization == 'elasticnet':
            l1_ratio = 0.5
            l1_contrib = self.reg_param * l1_ratio * np.sign(weights)
            l2_contrib = self.reg_param * (1 - l1_ratio) * weights
            return gradient + l1_contrib + l2_contrib

        return gradient

    def _compute_gradient(self, X_batch, y_batch):
        """Вычисление градиента для мини-батча"""
        m = X_batch.shape[0]
        predictions = X_batch @ self.weights + self.bias  # оптимизировано
        errors = predictions - y_batch

        # Оптимизированное вычисление градиента
        dw = (X_batch.T @ errors) / m
        db = np.mean(errors)

        # Применение регуляризации только к весам
        dw = self._apply_regularization(self.weights, dw)

        return dw, db

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Обучение модели на данных
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if X_val is not None:
            X_val = np.asarray(X_val)
            y_val = np.asarray(y_val)
            use_validation = True
        else:
            use_validation = False

        n_samples, n_features = X.shape
        self._initialize_parameters(n_features)

        self.loss_history = []
        self.val_loss_history = []
        no_improvement_count = 0

        for epoch in range(self.epochs):
            # Перемешиваем данные на каждой эпохе
            indices = self.rng.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Текущая скорость обучения
            current_lr = self._get_learning_rate(epoch)

            # Мини-батч SGD
            for i in range(0, n_samples, self.batch_size):
                end_idx = min(i + self.batch_size, n_samples)
                X_batch = X_shuffled[i:end_idx]
                y_batch = y_shuffled[i:end_idx]

                # Вычисление градиентов
                dw, db = self._compute_gradient(X_batch, y_batch)

                # Обновление с использованием импульса
                self.velocity_w = self.momentum * self.velocity_w - current_lr * dw
                self.velocity_b = self.momentum * self.velocity_b - current_lr * db

                # Обновление параметров
                self.weights += self.velocity_w
                self.bias += self.velocity_b

            # Расчет ошибки на обучающей выборке
            train_predictions = X @ self.weights + self.bias
            train_loss = np.mean((train_predictions - y) ** 2)
            self.loss_history.append(train_loss)

            # Расчет ошибки на валидационной выборке
            if use_validation:
                val_predictions = X_val @ self.weights + self.bias
                val_loss = np.mean((val_predictions - y_val) ** 2)
                self.val_loss_history.append(val_loss)

                # Сохраняем лучшие веса
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_weights = self.weights.copy()
                    self.best_bias = self.bias
                    self.best_epoch = epoch
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

            # Вывод прогресса
            if self.verbose and (epoch + 1) % 10 == 0:
                status = f"Эпоха {epoch + 1}/{self.epochs}, loss: {train_loss:.6f}"
                if use_validation:
                    status += f", val_loss: {val_loss:.6f}"
                print(status)

            # Ранняя остановка
            if self.early_stopping and use_validation and no_improvement_count >= self.patience:
                if self.verbose:
                    print(f"Ранняя остановка на эпохе {epoch + 1}")
                # Восстанавливаем лучшие веса
                self.weights = self.best_weights
                self.bias = self.best_bias
                break

        # Восстанавливаем лучшие веса в конце обучения
        if use_validation:
            self.weights = self.best_weights
            self.bias = self.best_bias

        return self

    def predict(self, X):
        """Выполнение прогноза"""
        X = np.asarray(X)
        return X @ self.weights + self.bias


def model_objective(trial, model):
    x, y = load_california(scale=True)

    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    batch_size = model.batch_size

    proc = psutil.Process()
    mem_before = proc.memory_info().rss

    t0 = time.perf_counter()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    t1 = time.perf_counter()

    mem_after = proc.memory_info().rss

    mse = mean_squared_error(y_test, y_pred)

    n_samples, n_features = X_train.shape
    n_batches_per_epoch = int(np.ceil(n_samples / batch_size))

    forward_ops = n_features + (n_features-1)

    gradient_ops = 2 * n_features

    weight_update_ops = 2 * (n_features + 1)  # +1 for bias

    ops_per_batch = batch_size * (forward_ops + gradient_ops) + weight_update_ops

    total_flops = ops_per_batch * n_batches_per_epoch * model.epochs

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
