from dataclasses import dataclass

import numpy as np


@dataclass
class MLPConfig:
    input_dim: int = 28 * 28
    hidden_units: tuple[int, int] = (256, 128)
    dropout: float = 0.2
    learning_rate: float = 1e-3
    batch_size: int = 128
    epochs: int = 10
    validation_split: float = 0.1
    seed: int = 7


def set_random_seed(seed):
    np.random.seed(seed)
    return np.random.default_rng(seed)


def clip_probabilities(probabilities, eps=1e-6):
    return np.clip(np.asarray(probabilities, dtype=float), eps, 1.0 - eps)


def logit(probabilities, eps=1e-6):
    clipped = clip_probabilities(probabilities, eps=eps)
    return np.log(clipped / (1.0 - clipped))


def reshape_heatmap(vector, shape):
    vector = np.asarray(vector, dtype=float)
    if int(np.prod(shape)) != vector.size:
        return vector.reshape(1, -1)
    return vector.reshape(shape)


def top_k_indices(values, top_k):
    values = np.asarray(values, dtype=float)
    top_k = max(1, min(top_k, values.size))
    indices = np.argpartition(np.abs(values), -top_k)[-top_k:]
    return indices[np.argsort(np.abs(values[indices]))[::-1]]


def classification_rate(probabilities, class_index):
    predicted = np.argmax(probabilities, axis=1)
    return float(np.mean(predicted == class_index))


def bootstrap_ci(values, alpha=0.05):
    data = np.asarray(list(values), dtype=float)
    if data.size == 0:
        return (float("nan"), float("nan"))
    lower = float(np.quantile(data, alpha / 2.0))
    upper = float(np.quantile(data, 1.0 - alpha / 2.0))
    return lower, upper


def load_mnist_data(flatten=True):
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError("TensorFlow is required to load MNIST via tf.keras.datasets.") from exc

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    if flatten:
        x_train = x_train.reshape((x_train.shape[0], -1))
        x_test = x_test.reshape((x_test.shape[0], -1))
    return {
        "x_train": x_train,
        "y_train": y_train.astype("int64"),
        "x_test": x_test,
        "y_test": y_test.astype("int64"),
    }


def stratified_subsample(x, y, sample_size, random_state=7):
    rng = np.random.default_rng(random_state)
    unique_classes = np.unique(y)
    per_class = max(1, sample_size // max(1, unique_classes.size))
    indices = []
    for class_id in unique_classes:
        class_indices = np.flatnonzero(y == class_id)
        draw_size = min(per_class, class_indices.size)
        indices.append(rng.choice(class_indices, size=draw_size, replace=False))
    merged = np.concatenate(indices)
    if merged.size < sample_size:
        remaining = np.setdiff1d(np.arange(y.shape[0]), merged, assume_unique=False)
        extra = rng.choice(remaining, size=sample_size - merged.size, replace=False)
        merged = np.concatenate([merged, extra])
    rng.shuffle(merged)
    return x[merged], y[merged]


class TargetMLP:
    def __init__(self, config=None):
        self.config = config or MLPConfig()
        self.model = None

    def _require_tf(self):
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError("TensorFlow is required for TargetMLP.") from exc
        return tf

    def _build(self):
        tf = self._require_tf()
        tf.keras.utils.set_random_seed(self.config.seed)
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(self.config.input_dim,)),
                tf.keras.layers.Dense(self.config.hidden_units[0], activation="relu"),
                tf.keras.layers.Dropout(self.config.dropout),
                tf.keras.layers.Dense(self.config.hidden_units[1], activation="relu"),
                tf.keras.layers.Dropout(self.config.dropout),
                tf.keras.layers.Dense(10, activation="softmax"),
            ]
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(self, x_train, y_train, verbose=1):
        self.model = self._build()
        history = self.model.fit(
            x_train,
            y_train,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            validation_split=self.config.validation_split,
            verbose=verbose,
        )
        return {key: list(value) for key, value in history.history.items()}

    def predict_proba(self, x, batch_size=256):
        if self.model is None:
            raise RuntimeError("TargetMLP must be fitted before calling predict_proba.")
        return np.asarray(self.model.predict(x, batch_size=batch_size, verbose=0), dtype=float)

    def evaluate(self, x, y):
        if self.model is None:
            raise RuntimeError("TargetMLP must be fitted before evaluation.")
        loss, accuracy = self.model.evaluate(x, y, verbose=0)
        return {"loss": float(loss), "accuracy": float(accuracy)}


def train_target_mlp(x_train, y_train, config=None, verbose=1):
    model = TargetMLP(config=config)
    history = model.fit(x_train, y_train, verbose=verbose)
    return model, history


def prepare_class_logits(model, x, class_index, eps=1e-6):
    probabilities = model.predict_proba(x)
    positive_prob = clip_probabilities(probabilities[:, class_index], eps=eps)
    return positive_prob, logit(positive_prob, eps=eps)


class LinearLogitSurrogate:
    def __init__(self, ridge=1e-3, feature_shape=(28, 28)):
        self.ridge = float(ridge)
        self.feature_shape = feature_shape
        self.intercept_ = None
        self.coef_ = None

    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        x_aug = np.column_stack([np.ones(x.shape[0]), x])
        penalty = self.ridge * np.eye(x_aug.shape[1])
        penalty[0, 0] = 0.0
        solution = np.linalg.solve(x_aug.T @ x_aug + penalty, x_aug.T @ y)
        self.intercept_ = float(solution[0])
        self.coef_ = solution[1:]
        return self

    def predict_logit(self, x):
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Baseline surrogate must be fitted before prediction.")
        x = np.atleast_2d(np.asarray(x, dtype=float))
        return x @ self.coef_ + self.intercept_

    def explain_instance(self, x, top_k=20):
        if self.coef_ is None:
            raise RuntimeError("Baseline surrogate must be fitted before explanation.")
        contributions = np.asarray(x, dtype=float) * self.coef_
        top_indices = top_k_indices(contributions, top_k)
        return {
            "dominant_component": 0,
            "coefficients": self.coef_.copy(),
            "contributions": contributions,
            "top_indices": top_indices,
            "top_weights": self.coef_[top_indices],
            "top_contributions": contributions[top_indices],
            "heatmap": reshape_heatmap(self.coef_, self.feature_shape),
        }
