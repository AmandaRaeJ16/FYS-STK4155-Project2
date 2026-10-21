import numpy as np

class ScratchClassifier:
    """
    Feed-forward neural network for multiclass classification.

    Hidden layers use ReLU or Leaky ReLU.
    The output layer uses softmax.
    The cost function is multiclass cross-entropy.
    """

    def __init__(
        self,
        hidden_layers=(128,),
        activation="relu",
        n_inputs=784,
        n_classes=10,
        seed=42,
    ):
        if activation not in ("relu", "leaky_relu", "sigmoid"):
            raise ValueError(
                "activation must be 'relu', 'leaky_relu', or 'sigmoid'"
            )

        self.hidden_layers = tuple(hidden_layers)
        self.activation = activation
        self.n_classes = n_classes
        self.seed = seed

        rng = np.random.default_rng(seed)

        sizes = (n_inputs,) + self.hidden_layers + (n_classes,)

        self.weights = []
        self.biases = []

        for layer_index, (n_in, n_out) in enumerate(
            zip(sizes[:-1], sizes[1:])
        ):
            is_output_layer = layer_index == len(sizes) - 2

            if not is_output_layer and activation in ("relu", "leaky_relu"):
                # He initialization for ReLU-family activations.
                std = np.sqrt(2.0 / n_in)
            else:
                # Xavier-like initialization.
                std = np.sqrt(1.0 / n_in)

            self.weights.append(
                rng.normal(
                    loc=0.0,
                    scale=std,
                    size=(n_in, n_out),
                )
            )

            self.biases.append(
                np.zeros((1, n_out))
            )

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
        }

    def _activate(self, z):
        if self.activation == "relu":
            return np.maximum(z, 0.0)

        if self.activation == "leaky_relu":
            return np.where(z > 0.0, z, 0.01 * z)

        # Numerically stable sigmoid.
        return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))

    def _activation_gradient(self, z):
        if self.activation == "relu":
            return (z > 0.0).astype(float)

        if self.activation == "leaky_relu":
            return np.where(z > 0.0, 1.0, 0.01)

        activated = self._activate(z)
        return activated * (1.0 - activated)

    @staticmethod
    def _softmax(z):
        # Subtracting the row maximum prevents exponential overflow.
        shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    @staticmethod
    def _one_hot(y, n_classes):
        encoded = np.zeros((len(y), n_classes), dtype=np.float64)
        encoded[np.arange(len(y)), y] = 1.0
        return encoded

    def forward(self, X, cache=False):
        activation = X
        activations = [activation]
        pre_activations = []

        # Hidden layers.
        for layer_index in range(len(self.weights) - 1):
            z = activation @ self.weights[layer_index]
            z += self.biases[layer_index]

            activation = self._activate(z)

            pre_activations.append(z)
            activations.append(activation)

        # Output layer.
        output_index = len(self.weights) - 1

        z_output = activation @ self.weights[output_index]
        z_output += self.biases[output_index]

        probabilities = self._softmax(z_output)

        pre_activations.append(z_output)
        activations.append(probabilities)

        if cache:
            return probabilities, activations, pre_activations

        return probabilities

    @staticmethod
    def cross_entropy(probabilities, y):
        n = len(y)
        correct_class_probabilities = probabilities[
            np.arange(n),
            y,
        ]

        return -np.mean(
            np.log(np.clip(correct_class_probabilities, 1e-15, 1.0))
        )

    @staticmethod
    def accuracy(probabilities, y):
        predictions = np.argmax(probabilities, axis=1)
        return np.mean(predictions == y)

    def gradients(self, X, y, l2=0.0):
        probabilities, activations, pre_activations = self.forward(
            X,
            cache=True,
        )

        y_one_hot = self._one_hot(y, self.n_classes)

        # Softmax plus cross-entropy derivative.
        delta = (probabilities - y_one_hot) / len(X)

        d_weights = [None] * len(self.weights)
        d_biases = [None] * len(self.biases)

        for layer_index in range(len(self.weights) - 1, -1, -1):
            d_weights[layer_index] = (
                activations[layer_index].T @ delta
            )

            if l2 > 0.0:
                d_weights[layer_index] += (
                    2.0 * l2 * self.weights[layer_index]
                )

            d_biases[layer_index] = np.sum(
                delta,
                axis=0,
                keepdims=True,
            )

            if layer_index > 0:
                delta = delta @ self.weights[layer_index].T
                delta *= self._activation_gradient(
                    pre_activations[layer_index - 1]
                )

        return d_weights, d_biases

    def predict_proba(self, X):
        return self.forward(X)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def fit(
        self,
        X,
        y,
        X_val=None,
        y_val=None,
        optimizer="adam",
        learning_rate=0.001,
        epochs=20,
        batch_size=128,
        l2=0.0,
        verbose=True,
    ):
        if optimizer not in ("gd", "rmsprop", "adam"):
            raise ValueError(
                "optimizer must be 'gd', 'rmsprop', or 'adam'"
            )

        rng = np.random.default_rng(self.seed + 1000)

        parameters = self.weights + self.biases

        first_moment = [
            np.zeros_like(parameter)
            for parameter in parameters
        ]

        second_moment = [
            np.zeros_like(parameter)
            for parameter in parameters
        ]

        time_step = 0

        for epoch in range(epochs):
            order = rng.permutation(len(X))

            for start in range(0, len(X), batch_size):
                batch_indices = order[start:start + batch_size]

                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                d_weights, d_biases = self.gradients(
                    X_batch,
                    y_batch,
                    l2=l2,
                )

                gradients = d_weights + d_biases
                time_step += 1

                for parameter_index, (
                    parameter,
                    gradient,
                ) in enumerate(zip(parameters, gradients)):

                    if optimizer == "gd":
                        parameter -= learning_rate * gradient

                    elif optimizer == "rmsprop":
                        second_moment[parameter_index] = (
                            0.9 * second_moment[parameter_index]
                            + 0.1 * gradient**2
                        )

                        parameter -= (
                            learning_rate
                            * gradient
                            / (
                                np.sqrt(
                                    second_moment[parameter_index]
                                )
                                + 1e-8
                            )
                        )

                    else:
                        first_moment[parameter_index] = (
                            0.9 * first_moment[parameter_index]
                            + 0.1 * gradient
                        )

                        second_moment[parameter_index] = (
                            0.999 * second_moment[parameter_index]
                            + 0.001 * gradient**2
                        )

                        corrected_first_moment = (
                            first_moment[parameter_index]
                            / (1.0 - 0.9**time_step)
                        )

                        corrected_second_moment = (
                            second_moment[parameter_index]
                            / (1.0 - 0.999**time_step)
                        )

                        parameter -= (
                            learning_rate
                            * corrected_first_moment
                            / (
                                np.sqrt(corrected_second_moment)
                                + 1e-8
                            )
                        )

            train_probabilities = self.predict_proba(X)

            train_loss = self.cross_entropy(
                train_probabilities,
                y,
            )

            train_accuracy = self.accuracy(
                train_probabilities,
                y,
            )

            self.history["train_loss"].append(train_loss)
            self.history["train_accuracy"].append(train_accuracy)

            if X_val is not None and y_val is not None:
                val_probabilities = self.predict_proba(X_val)

                val_loss = self.cross_entropy(
                    val_probabilities,
                    y_val,
                )

                val_accuracy = self.accuracy(
                    val_probabilities,
                    y_val,
                )

                self.history["val_loss"].append(val_loss)
                self.history["val_accuracy"].append(val_accuracy)

                if verbose:
                    print(
                        f"Epoch {epoch + 1:3d}/{epochs}: "
                        f"train loss={train_loss:.4f}, "
                        f"train acc={train_accuracy:.4f}, "
                        f"val loss={val_loss:.4f}, "
                        f"val acc={val_accuracy:.4f}"
                    )
            elif verbose:
                print(
                    f"Epoch {epoch + 1:3d}/{epochs}: "
                    f"train loss={train_loss:.4f}, "
                    f"train acc={train_accuracy:.4f}"
                )

        return self