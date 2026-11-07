"""
scratch_classifier.py -- our own feed-forward classifier in numpy

Same conventions as scratch_ffnn.py (samples in rows, weights of shape (inputs, outputs),
feed-forward step a @ W + b), but with a softmax output layer and the multiclass
cross-entropy cost.

Contents
  ScratchClassifier   hidden layers with sigmoid, ReLU or leaky ReLU, softmax output,
                      cross-entropy cost, optional L1 or L2 penalty on the weights,
                      training with plain GD, RMSprop or Adam on minibatches.
                      hidden_layers=() gives a network with no hidden layers, which is
                      multinomial logistic regression (our own baseline).
  check_gradients     compares backpropagation with jax.grad and with central finite
                      differences for every parameter of small networks.
"""
import numpy as np

class ScratchClassifier:
    """
    Feed-forward neural network for multiclass classification.

    Hidden layers use sigmoid, ReLU or leaky ReLU (the same activation in every hidden
    layer). The output layer uses softmax and the cost function is the multiclass
    cross-entropy. With hidden_layers=() the network is multinomial logistic regression.
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

    def gradients(self, X, y, l2=0.0, penalty=None, lam=0.0):
        if penalty not in (None, "l1", "l2"):
            raise ValueError("penalty must be None, 'l1' or 'l2'")
        if l2 > 0.0:
            penalty, lam = "l2", l2

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

            if penalty == "l2" and lam > 0.0:
                d_weights[layer_index] += (
                    2.0 * lam * self.weights[layer_index]
                )
            elif penalty == "l1" and lam > 0.0:
                d_weights[layer_index] += (
                    lam * np.sign(self.weights[layer_index])
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
        penalty=None,
        lam=0.0,

    ):
        
        """Train on minibatches of batch_size with 'gd', 'rmsprop' or 'adam'.

        l2 is the strength of an L2 penalty (as used in the report); alternatively pass
        penalty='l1' or 'l2' together with lam. The per-epoch training and validation
        loss and accuracy are stored in self.history.
        """
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
                    penalty=penalty,
                    lam=lam,
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
    
# Gradient check

def check_gradients(X, y, hidden_layers_list=((), (8, 6)),
                    activations=("sigmoid", "relu", "leaky_relu"),
                    n_classes=10, seed=9, eps=1e-6):
    
    """Compare the backpropagation of ScratchClassifier with jax.grad and with central
    finite differences of the cross-entropy, for every weight and bias.

    X, y : a small batch of inputs and integer labels (keep it small: the finite
           differences need two forward passes per parameter).
    Returns {(hidden_layers, activation): (jax_ok, finite_differences_ok)}.

    LLM-assisted (level 2): Claude (claude.ai, November 2026) supplied this
    function, modelled on check_gradients in scratch_ffnn.py. We integrated it
    with our ScratchClassifier and call it in part e of the notebook. Verified
    by running it on 20 MNIST images for a network without hidden layers and
    for (8, 6) networks with sigmoid, ReLU and leaky ReLU: every weight and bias
    agrees with jax.grad and with central finite differences.
    """

    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)

    jax_act = {"sigmoid": lambda z: 1 / (1 + jnp.exp(-jnp.clip(z, -50, 50))),
               "relu": lambda z: jnp.maximum(z, 0),
               "leaky_relu": lambda z: jnp.where(z > 0, z, 0.01 * z)}
    Y = ScratchClassifier._one_hot(y, n_classes)
    results = {}
    for hidden in hidden_layers_list:
        # with no hidden layers the activation plays no role: check it once
        acts = activations if len(hidden) > 0 else activations[:1]
        for act in acts:
            net = ScratchClassifier(hidden, act, n_inputs=X.shape[1],
                                    n_classes=n_classes, seed=seed)

            def cost(weights, biases):
                a = X
                for j, (W, b) in enumerate(zip(weights, biases)):
                    z = a @ W + b
                    a = z if j == len(weights) - 1 else jax_act[act](z)
                return -jnp.mean(jnp.sum(Y * jax.nn.log_softmax(a, axis=1), axis=1))

            gW, gb = net.gradients(X, y)
            jW, jb = jax.grad(cost, argnums=(0, 1))(net.weights, net.biases)
            jax_ok = all(np.allclose(a, b, rtol=1e-6, atol=1e-10)
                         for a, b in zip(gW + gb, list(jW) + list(jb)))

            fd_ok = True
            for params, grads in ((net.weights, gW), (net.biases, gb)):
                for P, G in zip(params, grads):
                    for idx in np.ndindex(P.shape):
                        old = P[idx]
                        P[idx] = old + eps
                        lp = net.cross_entropy(net.forward(X), y)
                        P[idx] = old - eps
                        lm = net.cross_entropy(net.forward(X), y)
                        P[idx] = old
                        if not np.isclose(G[idx], (lp - lm) / (2 * eps),
                                          rtol=1e-4, atol=1e-8):
                            fd_ok = False
            n_par = sum(W.size + b.size for W, b in zip(net.weights, net.biases))
            label = act if len(hidden) > 0 else "none (logistic regression)"
            print(f"hidden layers {str(hidden):7s} activation {label:27s} "
                  f"{n_par:5d} parameters:  backprop == jax.grad: {jax_ok},  "
                  f"backprop == finite differences: {fd_ok}")
            results[(hidden, act)] = (jax_ok, fd_ok)
    return results
