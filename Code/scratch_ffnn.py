"""
scratch_ffnn.py -- our own feed-forward neural network in numpy

Conventions (week 41/42 exercises, Section 8.5 of the lecture notes):
  * samples are stored in the rows of the input, X has shape (n_samples, n_inputs)
  * the weight matrix of a layer has shape (inputs, outputs)
  * the feed-forward step of a layer is  z = a @ W + b,  a_next = f(z)

Contents
  activate, activation_grad      sigmoid, ReLU, leaky ReLU and linear, with derivatives
  ScratchFFNN                    regression network: configurable hidden layers and
                                 activation, linear output, MSE cost, manual backpropagation,
                                 training with plain GD, SGD+RMSprop or SGD+Adam
  check_gradients                compares backpropagation with jax.grad and with
                                 central finite differences for every parameter

The class knows nothing about how the targets were scaled: all losses stored during
training (history, final_*_mse_scaled, best_val_mse_scaled) are in the units of the
y that was passed to fit. Convert to original units outside the class.

Model selection (cross-validation over the hyperparameters) is done in the notebook;
fit() trains one network on one data set.
"""
import numpy as np

# Activation functions and their derivatives

LEAKY_SLOPE = 0.01


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def activate(z, name):
    if name == 'sigmoid': return sigmoid(z)
    if name == 'relu': return np.maximum(z, 0)
    if name == 'leaky_relu': return np.where(z > 0, z, LEAKY_SLOPE*z)
    if name == 'linear': return z
    raise ValueError(name)


def activation_grad(z, name):
    if name == 'sigmoid':
        a = sigmoid(z)
        return a*(1-a)
    if name == 'relu': return (z > 0).astype(float)
    if name == 'leaky_relu': return np.where(z > 0, 1.0, LEAKY_SLOPE)
    if name == 'linear': return np.ones_like(z)
    raise ValueError(name)


# The network

class ScratchFFNN:

    def __init__(self, hidden_layers=(50,), activation='sigmoid', seed=42, n_inputs=1):
        self.hidden_layers = tuple(hidden_layers)
        self.activation = activation
        self.seed = seed
        rg = np.random.default_rng(seed)
        sizes = (n_inputs,) + self.hidden_layers + (1,)
        self.weights = []
        self.biases = []
        for j, (din, dout) in enumerate(zip(sizes[:-1], sizes[1:])):
            std = np.sqrt((2 if activation in ('relu','leaky_relu') and j < len(sizes)-2 else 1)/din)
            self.weights.append(rg.normal(0, std, size=(din, dout)))
            self.biases.append(np.zeros((1, dout)))

    def n_params(self):
        return sum(W.size + b.size for W, b in zip(self.weights, self.biases))

    def forward(self, X, cache=False):
        a = X
        activations = [a]
        zs = []
        for j, (W,b) in enumerate(zip(self.weights, self.biases)):
            z = a @ W + b
            a = activate(z, 'linear' if j == len(self.weights)-1 else self.activation)
            zs.append(z)
            activations.append(a)
        return (a, activations, zs) if cache else a

    def gradients(self, X, y, penalty=None, lam=0.0):
        """Backpropagation with optional L1 or L2 weight regularization."""
        if penalty not in (None, 'l1', 'l2'):
            raise ValueError("penalty must be None, 'l1', or 'l2'")
        if lam < 0:
            raise ValueError("lam must be non-negative")
        pred, acts, zs = self.forward(X, cache=True)
        # MSE = mean((prediction - target)^2), n samples, one output.
        delta = 2*(pred-y)/pred.size  # linear output derivative is 1
        dWs = [None]*len(self.weights)
        dbs = [None]*len(self.biases)
        for j in range(len(self.weights)-1, -1, -1):
            dWs[j] = acts[j].T @ delta
            dbs[j] = np.sum(delta, axis=0, keepdims=True)
            if penalty == 'l1':
                dWs[j] += lam*np.sign(self.weights[j])
            elif penalty == 'l2':
                dWs[j] += 2*lam*self.weights[j]
            if j > 0:
                delta = (delta @ self.weights[j].T)*activation_grad(zs[j-1], self.activation)
        return dWs, dbs

    def fit(self, X, y, X_val=None, y_val=None, optimizer='adam', lr=0.01,
            epochs=500, batch_size=32, patience=None, restore_best=True,
            penalty=None, lam=0.0):
        """Train with 'gd' (full batch), 'rmsprop' or 'adam' (minibatches of batch_size).

        X_val, y_val : optional validation data. Without them the network simply trains
                       for `epochs` epochs (used for the final fits on all training data).
        patience     : None means no early stopping; otherwise stop after `patience` epochs
                       without improvement of the validation MSE.
        restore_best : if True (and validation data is given) the weights with the lowest
                       validation MSE are restored at the end; if False the last-epoch
                       weights are kept (used for cross-validation curves).
        The per-epoch losses are kept in self.history (units of the y passed in).
        """
        if optimizer not in ('gd','rmsprop','adam'): raise ValueError(optimizer)
        if penalty not in (None, 'l1', 'l2'):
            raise ValueError("penalty must be None, 'l1', or 'l2'")
        if lam < 0:
            raise ValueError("lam must be non-negative")
        has_val = X_val is not None
        rg = np.random.default_rng(self.seed+1000)
        params = self.weights + self.biases
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        t = 0
        best_loss = float('inf')
        best = None
        stale = 0
        bs = len(X) if optimizer == 'gd' else min(batch_size, len(X))
        self.updates_per_epoch = int(np.ceil(len(X)/bs))
        history = {'train_mse':[], 'val_mse':[]}
        self.diverged = False
        with np.errstate(over='ignore', invalid='ignore'):
            for epoch in range(epochs):
                # Full batch for GD, mini-batches for Adam and RMSprop.
                shuffled = rg.permutation(len(X)) if optimizer != 'gd' else np.arange(len(X))
                for start in range(0, len(X), bs):
                    ids = shuffled[start:start+bs]
                    gW, gb = self.gradients(X[ids], y[ids], penalty=penalty, lam=lam)
                    gradients = gW+gb
                    t += 1
                    for j, (param, g) in enumerate(zip(params, gradients)):
                        if optimizer == 'gd':
                            param -= lr*g
                        elif optimizer == 'rmsprop':
                            v[j] = 0.9*v[j] + 0.1*g*g
                            param -= lr*g/(np.sqrt(v[j])+1e-8)
                        else: # Adam with bias correction
                            m[j] = 0.9*m[j] + 0.1*g
                            v[j] = 0.999*v[j] + 0.001*g*g
                            mh = m[j]/(1-0.9**t)
                            vh = v[j]/(1-0.999**t)
                            param -= lr*mh/(np.sqrt(vh)+1e-8)
                tr_loss = np.mean((self.forward(X)-y)**2)
                history['train_mse'].append(tr_loss)
                va_loss = np.mean((self.forward(X_val)-y_val)**2) if has_val else 0.0
                if has_val:
                    history['val_mse'].append(va_loss)
                if not (np.isfinite(tr_loss) and np.isfinite(va_loss)) or tr_loss > 1e6:
                    self.diverged = True
                    break
                if has_val:
                    if va_loss < best_loss-1e-8:
                        best_loss = va_loss
                        if restore_best:
                            best = ([w.copy() for w in self.weights], [b.copy() for b in self.biases])
                        stale = 0
                    else:
                        stale += 1
                        if patience is not None and stale >= patience: break
        # keep the last-epoch losses too (needed to show overfitting)
        self.final_train_mse_scaled = history['train_mse'][-1]
        self.final_val_mse_scaled = history['val_mse'][-1] if has_val else np.nan
        if restore_best and best is not None:
            self.weights, self.biases = best
        self.history = history
        self.best_val_mse_scaled = best_loss if has_val else np.nan
        return self


# Gradient checks

def finite_diff_ok(net, X, y, eps=1e-6):
    """Central finite differences for every weight and bias of net."""
    gW, gb = net.gradients(X, y)
    for params, grads in ((net.weights, gW), (net.biases, gb)):
        for P, G in zip(params, grads):
            for idx in np.ndindex(P.shape):
                old = P[idx]
                P[idx] = old+eps; lp = np.mean((net.forward(X)-y)**2)
                P[idx] = old-eps; lm = np.mean((net.forward(X)-y)**2)
                P[idx] = old
                if not np.isclose(G[idx], (lp-lm)/(2*eps), rtol=1e-4, atol=1e-8):
                    return False
    return True


def jax_grad_ok(net, X, y):
    """Compare backpropagation with jax.grad of the same cost"""
    import jax, jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    jax_act = {'sigmoid': lambda z: 1/(1+jnp.exp(-jnp.clip(z, -50, 50))),
               'relu': lambda z: jnp.maximum(z, 0),
               'leaky_relu': lambda z: jnp.where(z > 0, z, LEAKY_SLOPE*z)}[net.activation]

    def cost(weights, biases):
        a = X
        for j, (W, b) in enumerate(zip(weights, biases)):
            z = a @ W + b
            a = z if j == len(weights)-1 else jax_act(z)
        return jnp.mean((a-y)**2)

    gW, gb = net.gradients(X, y)
    jW, jb = jax.grad(cost, argnums=(0, 1))(net.weights, net.biases)
    return all(np.allclose(a, b) for a, b in zip(gW+gb, list(jW)+list(jb)))


def check_gradients(X, y, hidden_layers=(8, 6, 4), activations=('sigmoid', 'relu', 'leaky_relu'),
                    seed=9, n_fd=10):
    """Check every parameter of a small multi-layer network for each activation.
    Finite differences use the first n_fd samples to keep the check fast."""
    results = {}
    for act in activations:
        net = ScratchFFNN(hidden_layers, act, seed=seed, n_inputs=X.shape[1])
        results[act] = (jax_grad_ok(net, X, y), finite_diff_ok(net, X[:n_fd], y[:n_fd]))
        print(f"{act:11s}  all {net.n_params()} parameters:  backprop == jax.grad: {results[act][0]},"
              f"  backprop == finite differences: {results[act][1]}")
    return results