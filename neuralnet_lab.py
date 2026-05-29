import math, random, sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

np.random.seed(42)
random.seed(42)

def plot_series(xs, ys, title=""):
    plt.figure()
    plt.plot(xs, ys)
    plt.xlabel("step/epoch")
    plt.ylabel("value")
    if title:
        plt.title(title)
    plt.show()

def assert_close(name, got, want, atol=1e-7, rtol=1e-5):
    ok = np.allclose(got, want, atol=atol, rtol=rtol)
    print(f"[{name}] match:", ok, f"(max abs err ~ {np.max(np.abs(got - want)) if got.shape == want.shape else 'shape mismatch'})")
    if not ok:
        print("   shapes:", got.shape, "vs", want.shape)

### Stage 1 ###
def stage1_linear_estimator():
    n = 200
    true_a, true_b = 2.5, -0.7
    x = np.linspace(-2, 2, n)
    noise = 0.25 * np.random.randn(n)
    y = true_a * x + true_b + noise
    w = np.random.randn()
    b = np.random.randn()
    lr = 0.05
    losses = []
    for step in range(800):
        yhat = w * x + b
        err = yhat - y
        loss = np.mean(err**2)
        dw = 2 * np.mean(err * x)
        db = 2 * np.mean(err)
        w -= lr * dw
        b -= lr * db
        losses.append(loss)
    print("Stage1 fit:", {"w": w, "b": b, "true_a": true_a, "true_b": true_b})
    plot_series(range(len(losses)), losses, title="MSE over steps (Stage 1)")

### Stage 2 ###
class Linear:
    def __init__(self, in_dim, out_dim):
        limit = math.sqrt(6/(in_dim+out_dim))
        self.W = np.random.uniform(-limit, limit, size=(out_dim, in_dim))
        self.b = np.zeros((out_dim,))
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.x = None

    def forward(self, x):
        self.x = np.atleast_2d(np.asarray(x))
        out = self.x @ self.W.T + self.b
        return out

    def backward(self, dout):
        dout = np.atleast_2d(np.asarray(dout))
        x = np.atleast_2d(np.asarray(self.x))
        self.dW = dout.T @ x / x.shape[0]
        self.db = np.mean(dout, axis=0)
        dx = dout @ self.W
        return dx

def stage2_two_layer_linear():
    n = 200
    true_a, true_b = 2.5, -0.7
    x = np.linspace(-2, 2, n)
    noise = 0.25 * np.random.randn(n)
    y = true_a * x + true_b + noise
    X = x.reshape(-1, 1)
    Y = y.reshape(-1, 1)
    lin1 = Linear(in_dim=1, out_dim=4)
    lin2 = Linear(in_dim=4, out_dim=1)
    lr = 0.05
    losses = []
    for epoch in range(200):
        h = lin1.forward(X)
        yhat = lin2.forward(h)
        err = yhat - Y
        loss = np.mean(err**2)
        losses.append(loss)
        bs = X.shape[0]
        dyhat = (2.0/bs) * err
        dh = lin2.backward(dyhat)
        _ = lin1.backward(dh)
        for layer in (lin1, lin2):
            layer.W -= lr * layer.dW
            layer.b -= lr * layer.db
    plot_series(range(len(losses)), losses, title="Two-layer linear: MSE")
    print("Final loss:", losses[-1])

### Stage 3 ###
def relu(x):
    x = np.asarray(x)
    return np.maximum(0, x)

def relu_backward(x, dout):
    dx = dout * (x > 0).astype(float)
    return dx

def tanh(x):
    return np.tanh(x)

def tanh_backward(x, dout):
    t = np.tanh(x)
    dx = dout * (1.0 - t*t)
    return dx

class MLP:
    def __init__(self, in_dim, hidden_dim, out_dim, activation='relu'):
        self.lin1 = Linear(in_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, out_dim)
        self.activation = activation
        self.hidden_pre = None
        self.hidden = None

    def forward(self, X):
        z = self.lin1.forward(X)
        self.hidden_pre = z
        if self.activation == 'relu':
            self.hidden = relu(z)
        elif self.activation == 'tanh':
            self.hidden = tanh(z)
        out = self.lin2.forward(self.hidden)
        return out

    def backward(self, dout):
        dh = self.lin2.backward(dout)
        if self.activation == 'relu':
            dz = relu_backward(self.hidden_pre, dh)
        elif self.activation == 'tanh':
            dz = tanh_backward(self.hidden_pre, dh)
        _ = self.lin1.backward(dz)

def stage3_mlp_toy():
    n = 400
    X = np.linspace(-2, 2, n).reshape(-1,1)
    Y = np.sin(X)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    mlp = MLP(in_dim=1, hidden_dim=16, out_dim=1, activation='relu')
    lr = 0.1
    losses = []
    for epoch in range(800):
        yhat = mlp.forward(Xs)
        err = yhat - Y
        loss = np.mean(err**2)
        losses.append(loss)
        dy = (2.0/ Xs.shape[0]) * err
        mlp.backward(dy)
        for layer in (mlp.lin1, mlp.lin2):
            layer.W -= lr * layer.dW
            layer.b -= lr * layer.db
    plot_series(range(len(losses)), losses, title="MLP toy regression loss")
    print("Final MSE (MLP toy):", losses[-1])

def grad_check_linear():
    layer = Linear(in_dim=3, out_dim=2)
    X = np.random.randn(4,3)
    eps = 1e-5
    out = layer.forward(X)
    loss = np.sum(out**2)
    dloss_dout = 2 * out
    layer.backward(dloss_dout)
    analytic_dW = layer.dW.copy()
    analytic_db = layer.db.copy()
    numeric_dW = np.zeros_like(layer.W)
    for i in range(layer.W.shape[0]):
        for j in range(layer.W.shape[1]):
            orig = layer.W[i,j]
            layer.W[i,j] = orig + eps
            out1 = layer.forward(X); L1 = np.sum(out1**2)
            layer.W[i,j] = orig - eps
            out2 = layer.forward(X); L2 = np.sum(out2**2)
            numeric_dW[i,j] = (L1 - L2) / (2*eps)
            layer.W[i,j] = orig
    numeric_db = np.zeros_like(layer.b)
    for i in range(layer.b.shape[0]):
        orig = layer.b[i]
        layer.b[i] = orig + eps
        out1 = layer.forward(X); L1 = np.sum(out1**2)
        layer.b[i] = orig - eps
        out2 = layer.forward(X); L2 = np.sum(out2**2)
        numeric_db[i] = (L1 - L2) / (2*eps)
        layer.b[i] = orig
    print("Max abs dW error:", np.max(np.abs(analytic_dW - numeric_dW)))
    print("Max abs db error:", np.max(np.abs(analytic_db - numeric_db)))

### Stage 4 ###
def stage4_credit_score():
    np.random.seed(0)
    n = 500
    income = np.random.normal(50, 15, size=n)
    age = np.random.normal(40, 12, size=n)
    debt = np.random.normal(10, 6, size=n)
    w_true = np.array([0.05, -0.02, 0.1])
    b_true = -2.0
    X = np.vstack([income, age, debt]).T
    logits = X @ w_true + b_true
    prob = 1/(1+np.exp(-logits))
    y = (np.random.rand(n) < prob).astype(float).reshape(-1,1)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    W = np.random.randn(3,1) * 0.01
    b = np.zeros((1,))
    lr = 0.5
    lamb = 0.1
    losses = []
    for epoch in range(500):
        logits = Xs @ W + b
        preds = 1/(1+np.exp(-logits))
        eps = 1e-9
        loss = -np.mean(y * np.log(preds+eps) + (1-y) * np.log(1-preds+eps)) + 0.5 * lamb * np.sum(W**2)
        losses.append(loss)
        dlogits = (preds - y) / Xs.shape[0]
        dW = Xs.T @ dlogits + lamb * W
        db = np.mean(dlogits, axis=0)
        W -= lr * dW
        b -= lr * db
    plot_series(range(len(losses)), losses, title="Credit-score logistic loss (with L2)")
    print("Learned weights (unscaled):", W.ravel())
    sigma = scaler.scale_
    weight_original = (W.ravel() / sigma)
    print("Estimated effect per original feature (income, age, debt):", weight_original)
    print("Interpretation: positive -> increases default probability, negative -> decreases")

### Stage 5 ###
def softmax(z):
    z = np.asarray(z)
    zmax = np.max(z, axis=1, keepdims=True)
    exp = np.exp(z - zmax)
    return exp / np.sum(exp, axis=1, keepdims=True)

def cross_entropy_loss(probs, y_onehot):
    eps = 1e-9
    return -np.mean(np.sum(y_onehot * np.log(probs + eps), axis=1))

def stage5_digits_classifier(hidden_dim=64, epochs=50, batch_size=64, lr=0.1):
    digits = load_digits()
    X = digits.data
    y = digits.target
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    Xtrain, Xtest, ytrain, ytest = train_test_split(Xs, y, test_size=0.2, random_state=42, stratify=y)
    encoder = OneHotEncoder(sparse_output=False)
    Ytrain = encoder.fit_transform(ytrain.reshape(-1,1))
    Ytest = encoder.transform(ytest.reshape(-1,1))
    n_in = Xtrain.shape[1]
    n_out = Ytrain.shape[1]
    mlp = MLP(in_dim=n_in, hidden_dim=hidden_dim, out_dim=n_out, activation='relu')
    train_losses = []
    train_accs = []
    test_accs = []
    for epoch in range(epochs):
        idx = np.random.permutation(Xtrain.shape[0])
        for i in range(0, Xtrain.shape[0], batch_size):
            batch_idx = idx[i:i+batch_size]
            xb = Xtrain[batch_idx]
            yb = Ytrain[batch_idx]
            logits = mlp.forward(xb)
            probs = softmax(logits)
            loss = cross_entropy_loss(probs, yb)
            dlogits = (probs - yb) / xb.shape[0]
            mlp.backward(dlogits)
            for layer in (mlp.lin1, mlp.lin2):
                layer.W -= lr * layer.dW
                layer.b -= lr * layer.db
        train_logits = mlp.forward(Xtrain)
        train_probs = softmax(train_logits)
        train_loss = cross_entropy_loss(train_probs, Ytrain)
        ypred_train = np.argmax(train_probs, axis=1)
        train_acc = np.mean(ypred_train == ytrain)
        test_logits = mlp.forward(Xtest)
        test_probs = softmax(test_logits)
        ypred_test = np.argmax(test_probs, axis=1)
        test_acc = np.mean(ypred_test == ytest)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        if (epoch+1) % 5 == 0 or epoch==0:
            print(f"Epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f} train_acc={train_acc:.3f} test_acc={test_acc:.3f}")
    plot_series(range(len(train_losses)), train_losses, title="Digits: cross-entropy loss")
    print("Final test accuracy:", test_accs[-1])

### Main ###
def main():
    print("Stage 1 (Linear estimator) - running...")
    stage1_linear_estimator()

    print("Stage 2 (two-layer linear) - running...")
    stage2_two_layer_linear()

    print("Gradient-check for Linear layer...")
    grad_check_linear()

    print("Stage 3 (MLP toy regression)...")
    stage3_mlp_toy()

    print("Stage 4 (credit-score toy w/ L2)...")
    stage4_credit_score()

    print("Stage 5 (digits classifier)...")
    stage5_digits_classifier(hidden_dim=64, epochs=50, batch_size=64, lr=0.1)

if __name__ == "__main__":
    main()
