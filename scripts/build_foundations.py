"""Foundations and neural-net notebooks."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from nbutil import BOOT, colab, make_notebook  # noqa: E402


def check(cells):
    for kind, source in cells:
        if kind != "code":
            continue
        cleaned = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("%", "!")))
        ast.parse(cleaned)


def emit(path, cells, meta):
    check(cells)
    make_notebook(path, cells, meta)


def supervised():
    cells = [
        ("md", """
# Supervised learning

**Track:** Foundations. **Level:** Foundations. **CPU:** yes.

## Goals

- Fit a classifier and a regressor and report the metric that matches the decision.
- See why accuracy is the wrong summary on an imbalanced problem.
- Estimate generalization with a held-out split and with cross-validation.
- Produce a model that memorizes the training set.

## Sources and license

Original notebook following the scikit-learn supervised tutorials (BSD-3): classification metrics, cross-validation, and the overfitting discussion in the user guide. The API calls follow that documentation. The datasets are synthetic, from `sklearn.datasets`.

## Concepts

A supervised model sees pairs `(x, y)` and is scored on pairs it did not fit. Accuracy is the fraction of labels that match. Precision is the fraction of predicted positives that are positive. Recall is the fraction of real positives that were found. F1 is their harmonic mean. ROC-AUC ranks scores and does not need a threshold. For regression, mean squared error is in units of `y` squared and R² compares the model with predicting the mean. Cross-validation averages a metric over folds so one lucky split is not the whole story. Overfitting is a training score that is far better than the test score.
"""),
        ("code", colab(["numpy", "pandas", "scikit-learn", "matplotlib"])),
        ("code", BOOT + """
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from labs._shared.seed import seed_everything

seed_everything(0)
"""),
        ("md", """
## Guided exercise: imbalanced classification

Ninety-five percent of the rows are the negative class. A classifier that always says negative looks accurate and never finds a positive.
"""),
        ("code", r'''
X, y = make_classification(
    n_samples=1200,
    n_features=12,
    n_informative=6,
    n_redundant=2,
    weights=[0.95, 0.05],
    class_sep=1.2,
    flip_y=0.01,
    random_state=0,
)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
scaler = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

majority = DummyClassifier(strategy="most_frequent").fit(Xtr_s, ytr)
logit = LogisticRegression(max_iter=500).fit(Xtr_s, ytr)

def classification_row(name, model):
    pred = model.predict(Xte_s)
    proba = model.predict_proba(Xte_s)[:, 1] if hasattr(model, "predict_proba") else pred
    return {
        "model": name,
        "accuracy": accuracy_score(yte, pred),
        "precision": precision_score(yte, pred, zero_division=0),
        "recall": recall_score(yte, pred, zero_division=0),
        "f1": f1_score(yte, pred, zero_division=0),
        "roc_auc": roc_auc_score(yte, proba),
    }

table = pd.DataFrame([
    classification_row("majority class", majority),
    classification_row("logistic regression", logit),
])
print(table.round(3).to_string(index=False))
assert table.loc[0, "accuracy"] > 0.9
assert table.loc[0, "f1"] < table.loc[1, "f1"]
assert table.loc[1, "roc_auc"] > 0.7

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
cv = cross_val_score(LogisticRegression(max_iter=500), scaler.transform(X), y, cv=folds, scoring="f1")
print("5-fold F1", np.round(cv, 3), "mean", round(float(cv.mean()), 3))
assert cv.mean() > table.loc[0, "f1"]
'''),
        ("md", """
## Guided exercise: regression

Ridge regression on a synthetic linear target with noise. Mean squared error and R² are both reported. R² near 1 means the residual is small relative to the variance of `y`. It does not mean the model is useful outside this distribution.
"""),
        ("code", r'''
Xr, yr = make_regression(n_samples=400, n_features=6, noise=15.0, random_state=0)
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, test_size=0.25, random_state=0)
reg = Ridge(alpha=1.0).fit(Xr_tr, yr_tr)
pred = reg.predict(Xr_te)
mse = mean_squared_error(yr_te, pred)
r2 = r2_score(yr_te, pred)
print(f"test MSE {mse:.2f}  R2 {r2:.3f}")
assert r2 > 0.8
'''),
        ("md", """
## Your turn

Fit a depth-1 decision tree and an unrestricted tree on a small, wide training set. Compare training accuracy, test accuracy, and the gap between them.

## Solution
"""),
        ("code", r'''
Xo, yo = make_classification(
    n_samples=180, n_features=30, n_informative=6, n_redundant=0, random_state=1,
)
Xo_tr, Xo_te, yo_tr, yo_te = train_test_split(Xo, yo, test_size=0.4, random_state=1)
rows = []
for depth, model in (
    (1, DecisionTreeClassifier(max_depth=1, random_state=0)),
    (None, DecisionTreeClassifier(max_depth=None, random_state=0)),
):
    model.fit(Xo_tr, yo_tr)
    train_acc = model.score(Xo_tr, yo_tr)
    test_acc = model.score(Xo_te, yo_te)
    rows.append({"max_depth": str(depth), "train": train_acc, "test": test_acc, "gap": train_acc - test_acc})
overfit = pd.DataFrame(rows)
print(overfit.round(3).to_string(index=False))
assert overfit.loc[1, "train"] > overfit.loc[0, "train"]
assert overfit.loc[1, "gap"] > overfit.loc[0, "gap"]
'''),
        ("md", """
## Beyond the course

Sweep the decision threshold on the logistic scores. A higher threshold predicts fewer positives. Precision and recall are not required to move in lockstep on a tiny test set, so the recorded result is the positive rate, plus the precision and recall at each threshold. The question underneath is which threshold the product can live with.
"""),
        ("code", r'''
scores = logit.predict_proba(Xte_s)[:, 1]
sweep = []
for threshold in (0.2, 0.5, 0.8):
    pred = (scores >= threshold).astype(int)
    sweep.append({
        "threshold": threshold,
        "predicted_positive_rate": float(pred.mean()),
        "precision": precision_score(yte, pred, zero_division=0),
        "recall": recall_score(yte, pred, zero_division=0),
    })
sweep_df = pd.DataFrame(sweep)
print(sweep_df.round(3).to_string(index=False))
assert sweep_df["predicted_positive_rate"].is_monotonic_decreasing

fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(sweep_df["threshold"], sweep_df["precision"], marker="o", label="precision")
ax.plot(sweep_df["threshold"], sweep_df["recall"], marker="o", label="recall")
ax.set_xlabel("threshold")
ax.legend()
ax.set_title("Imbalanced test set")
plt.tight_layout()
plt.show()
print("LAB OK")
'''),
        ("md", """
## Research questions

1. The majority classifier has high accuracy and zero F1. Name a product decision for which you would still ship the majority classifier, and one for which you would not.
2. Five-fold F1 moved between folds. How large a gap between folds would stop you from quoting the mean alone?
3. The unrestricted tree's training accuracy is not evidence about the test set. What would you change in the data, rather than in `max_depth`, to shrink that gap?

## Further reading

- scikit-learn user guide, sections "Supervised learning", "Model evaluation", and "Cross-validation" (BSD-3). https://scikit-learn.org/stable/supervised_learning.html
- Hastie, Tibshirani, and Friedman, *The Elements of Statistical Learning*, chapters 2 and 7. Reference, not copied.
"""),
    ]
    emit(
        "labs/foundations/01_supervised_learning/supervised_learning.ipynb",
        cells,
        {"track": "foundations", "lab": "supervised", "needs_gpu": False, "needs_api_key": False},
    )


def unsupervised():
    cells = [
        ("md", """
# Unsupervised learning

**Track:** Foundations. **Level:** Foundations. **CPU:** yes.

## Goals

- Cluster a labeled blob dataset and score the clusters against the labels you are not allowed to train on.
- Compress a wide matrix with PCA and check that more components reconstruct it better.
- Embed the same points in two dimensions with UMAP.
- Flag planted outliers with an isolation forest.

## Sources and license

Original notebook following the scikit-learn clustering, PCA, and outlier tutorials (BSD-3) and the UMAP paper and library (BSD-3, McInnes, Healy, and Melville). UMAP is called as a library. Its source is not copied. Labels are used only for scoring.

## Concepts

K-means minimizes squared distance to `k` centers. It does not know the true labels. Adjusted Rand index compares two partitions and is 1 when they match up to a permutation of the names. PCA finds orthogonal directions of variance. The reconstruction from `k` components is the best rank-`k` approximation in squared error. UMAP places neighbors near each other in a lower dimension. It is a visualization tool here, not a classifier. Isolation Forest scores how easy a point is to separate with random cuts. Easy-to-isolate points are the candidates for anomalies.
"""),
        ("code", colab(["numpy", "pandas", "scikit-learn", "matplotlib", "umap-learn"])),
        ("code", BOOT + """
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
from sklearn.ensemble import IsolationForest
from sklearn.metrics import adjusted_rand_score

from labs._shared.seed import seed_everything

seed_everything(0)
rng = np.random.default_rng(0)
"""),
        ("md", """
## Guided exercise: k-means and PCA

The blobs are generated with three centers. K-means is not given that fact as a feature. It is given `n_clusters=3` as a choice you made. PCA is fit on a 10-dimensional matrix whose first three coordinates were stretched, so a few components should explain most of the variance.
"""),
        ("code", r'''
Xb, yb = make_blobs(n_samples=400, centers=3, n_features=2, cluster_std=0.55, random_state=0)
labels = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(Xb)
ari = adjusted_rand_score(yb, labels)
print("adjusted Rand", round(ari, 3))
assert ari > 0.9

fig, ax = plt.subplots(figsize=(4, 3))
ax.scatter(Xb[:, 0], Xb[:, 1], c=labels, s=12, cmap="tab10")
ax.set_title(f"k-means, ARI {ari:.2f}")
plt.tight_layout()
plt.show()

Xw = rng.normal(size=(250, 10))
Xw[:, :3] *= 4.0
pca = PCA(random_state=0).fit(Xw)
print("explained variance ratio", np.round(pca.explained_variance_ratio_, 3))
assert np.isclose(pca.explained_variance_ratio_.sum(), 1.0)
assert pca.explained_variance_ratio_[:3].sum() > 0.8

def reconstruction_mse(components: int) -> float:
    model = PCA(n_components=components, random_state=0).fit(Xw)
    restored = model.inverse_transform(model.transform(Xw))
    return float(np.mean((Xw - restored) ** 2))

mse2, mse6 = reconstruction_mse(2), reconstruction_mse(6)
print("reconstruction MSE k=2", round(mse2, 3), "k=6", round(mse6, 3))
assert mse6 < mse2
'''),
        ("md", """
## Guided exercise: UMAP and outliers

UMAP is fit with `compatibility_layout=True` so the embedding matches the library's previous layout algorithm. The assertion is the output shape and a finite embedding, not a particular picture. The outlier set is 300 inliers from a standard normal and 25 points shifted far away. Precision is the fraction of isolation-forest flags that are real planted outliers.
"""),
        ("code", r'''
embedding = umap.UMAP(
    n_components=2, n_neighbors=15, min_dist=0.1, random_state=0, compatibility_layout=True,
).fit_transform(Xb)
print("UMAP shape", embedding.shape)
assert embedding.shape == (len(Xb), 2)
assert np.isfinite(embedding).all()

fig, ax = plt.subplots(figsize=(4, 3))
ax.scatter(embedding[:, 0], embedding[:, 1], c=yb, s=12, cmap="tab10")
ax.set_title("UMAP of the blobs")
plt.tight_layout()
plt.show()

inliers = rng.normal(size=(300, 6))
outliers = rng.normal(loc=6.0, size=(25, 6))
Xo = np.vstack([inliers, outliers])
truth = np.array([0] * len(inliers) + [1] * len(outliers))
flags = IsolationForest(n_estimators=200, contamination=0.08, random_state=0).fit_predict(Xo) == -1
precision = float(truth[flags].mean()) if flags.any() else 0.0
recall = float(flags[truth == 1].mean())
print(f"outlier precision {precision:.2f}  recall {recall:.2f}  flags {int(flags.sum())}")
assert precision > 0.7
print("LAB OK")
'''),
        ("md", """
## Your turn

Plot k-means inertia for `k` from 1 to 6 on the blob data. Inertia must fall as `k` rises. That plot does not tell you the true number of clusters by itself. The labels did.

## Solution
"""),
        ("code", r'''
inertias = []
for k in range(1, 7):
    inertias.append(KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xb).inertia_)
print(pd.DataFrame({"k": range(1, 7), "inertia": inertias}).round(1).to_string(index=False))
assert all(a > b for a, b in zip(inertias, inertias[1:]))
'''),
        ("md", """
## Beyond the course

Refit UMAP at `n_neighbors` 5 and 50 and report the mean distance from each point to its nearest other point in the embedding. A smaller neighborhood parameter is allowed to fragment the picture. The measurement is the comparison. It is not a claim that one picture is the data.
"""),
        ("code", r'''
def mean_nearest(z):
    d = np.sqrt(((z[:, None, :] - z[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(d, np.inf)
    return float(d.min(axis=1).mean())

for neighbors in (5, 50):
    z = umap.UMAP(
        n_components=2, n_neighbors=neighbors, min_dist=0.1, random_state=0, compatibility_layout=True,
    ).fit_transform(Xb)
    print(f"n_neighbors {neighbors} mean nearest-neighbor distance {mean_nearest(z):.3f}")
'''),
        ("md", """
## Research questions

1. Adjusted Rand used the blob labels. What can you monitor in production when you do not have them?
2. PCA reconstruction error fell with more components. When is a lower reconstruction error a worse representation for a downstream classifier?
3. The isolation forest was told a contamination rate close to the true outlier rate. What happens to precision if you pass 0.01 or 0.3 instead?

## Further reading

- scikit-learn user guide, clustering, PCA, and novelty/outlier detection (BSD-3).
- McInnes, Healy, and Melville, *UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction* (2018). Library license BSD-3.
- Liu, Ting, and Zhou, *Isolation Forest* (ICDM 2008).
"""),
    ]
    emit(
        "labs/foundations/02_unsupervised_learning/unsupervised_learning.ipynb",
        cells,
        {"track": "foundations", "lab": "unsupervised", "needs_gpu": False, "needs_api_key": False},
    )


def neural():
    cells = [
        ("md", """
# Neural nets from scratch

**Track:** Training. **Level:** Foundations. **CPU:** yes.

## Goals

- Implement a scalar autograd engine and check a gradient against a finite difference.
- Train a small multilayer perceptron with that engine.
- Train a PyTorch network on the same kind of problem, plot the loss, and apply weight decay.

## Sources and license

Original implementation. The `Value` API follows Andrej Karpathy's micrograd (MIT): addition, multiplication, `tanh`, and a reverse topological walk. The micrograd source file is not copied. The PyTorch loop follows the training pattern in the PyTorch tutorials (BSD-style) and the "from scratch, then a framework" order used by fast.ai (Apache-2.0 course code, not copied) and by Karpathy's Neural Networks: Zero to Hero (MIT).

## Concepts

Backpropagation applies the chain rule on the computation graph. Each node stores the local derivative of its output with respect to its inputs. A reverse walk multiplies those local derivatives by the upstream gradient. A finite difference `(f(x+ε) - f(x-ε)) / (2ε)` is the check that the local derivatives were written down correctly. A training loop is then: forward, loss, backward, step, zero the gradients. Weight decay adds a penalty on the size of the weights. Dropout and weight decay are regularizers. This notebook uses weight decay because its effect on the weight norm is measurable in a few steps.
"""),
        ("code", colab(["numpy", "torch", "scikit-learn", "matplotlib"])),
        ("code", BOOT + """
%matplotlib inline
import math
import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.datasets import make_moons

from labs._shared.seed import seed_everything

seed_everything(0)
torch.manual_seed(0)
"""),
        ("md", """
## Guided exercise: reverse-mode autodiff

`Value` records children and a closure that adds the local contribution into `grad`. `backward` walks the graph in an order where every child is finished before its parents are asked to push a gradient. The finite-difference check is on `tanh(x²)` at `x = 1.5`.
"""),
        ("code", r'''
class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, exponent):
        out = Value(self.data ** exponent, (self,), f"**{exponent}")

        def _backward():
            self.grad += exponent * (self.data ** (exponent - 1)) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1.0 - t * t) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo, seen = [], set()

        def build(node):
            if node not in seen:
                seen.add(node)
                for child in node._prev:
                    build(child)
                topo.append(node)

        build(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other if isinstance(other, Value) else -other)

    def __rmul__(self, other):
        return self * other

x = Value(1.5)
y = (x * x).tanh()
y.backward()
eps = 1e-5

def f(v):
    return math.tanh(v * v)

numerical = (f(1.5 + eps) - f(1.5 - eps)) / (2 * eps)
print("analytic", x.grad, "numeric", numerical)
assert abs(x.grad - numerical) < 1e-4
'''),
        ("md", """
## Guided exercise: a tiny trained network

The targets are linearly separable on purpose. The point of this cell is the training loop on the scalar engine: forward, squared error, `backward`, a gradient step, and an explicit zeroing of every `grad`. The PyTorch section is where the model has to fit a curve.
"""),
        ("code", r'''
class Neuron:
    def __init__(self, n_in):
        self.w = [Value(random.uniform(-1.0, 1.0)) for _ in range(n_in)]
        self.b = Value(0.0)

    def __call__(self, x):
        act = self.b
        for wi, xi in zip(self.w, x):
            act = act + wi * xi
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, n_in, n_out):
        self.neurons = [Neuron(n_in) for _ in range(n_out)]

    def __call__(self, x):
        out = [neuron(x) for neuron in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    def __init__(self, sizes):
        self.layers = [Layer(a, b) for a, b in zip(sizes, sizes[1:])]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


points = [((random.uniform(-2, 2), random.uniform(-2, 2))) for _ in range(24)]
targets = [1.0 if p[0] > 0 else -1.0 for p in points]
net = MLP([2, 8, 1])
losses = []
for step in range(40):
    total = Value(0.0)
    correct = 0
    for (x0, x1), target in zip(points, targets):
        pred = net([Value(x0), Value(x1)])
        total = total + (pred - target) ** 2
        correct += int((pred.data > 0) == (target > 0))
    loss = total * (1.0 / len(points))
    for p in net.parameters():
        p.grad = 0.0
    loss.backward()
    for p in net.parameters():
        p.data += -0.2 * p.grad
    losses.append(loss.data)
    if step in (0, 39):
        print(f"step {step} loss {loss.data:.3f} acc {correct / len(points):.2f}")
assert losses[-1] < losses[0]
'''),
        ("md", """
## Guided exercise: PyTorch training loop

Two-moons is not linearly separable. The network is a two-hidden-layer MLP. One run uses AdamW with `weight_decay=0`. The other uses the same learning rate and `weight_decay=0.05`. Both should drive the training loss down. The decayed run should finish with a smaller parameter norm. Validation accuracy is printed. It is not the thing the assertion treats as a leaderboard.
"""),
        ("code", r'''
class TorchMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def run(weight_decay: float, steps: int = 120):
    torch.manual_seed(0)
    Xm, ym = make_moons(n_samples=240, noise=0.18, random_state=0)
    X = torch.tensor(Xm, dtype=torch.float32)
    y = torch.tensor(ym, dtype=torch.float32).view(-1, 1)
    n = int(0.75 * len(X))
    model = TorchMLP()
    opt = torch.optim.AdamW(model.parameters(), lr=0.04, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    train_losses, val_losses = [], []
    for _ in range(steps):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(X[:n]), y[:n])
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            train_losses.append(float(loss_fn(model(X[:n]), y[:n])))
            val_losses.append(float(loss_fn(model(X[n:]), y[n:])))
    with torch.no_grad():
        acc = float(((model(X[n:]) > 0).float() == y[n:]).float().mean())
        norm = float(torch.sqrt(sum(p.detach().pow(2).sum() for p in model.parameters())))
    return train_losses, val_losses, acc, norm

free_train, free_val, free_acc, free_norm = run(0.0)
decay_train, decay_val, decay_acc, decay_norm = run(0.05)
print(f"no decay  val acc {free_acc:.2f}  weight norm {free_norm:.2f}  final train loss {free_train[-1]:.3f}")
print(f"decay     val acc {decay_acc:.2f}  weight norm {decay_norm:.2f}  final train loss {decay_train[-1]:.3f}")
assert free_train[-1] < free_train[0]
assert decay_train[-1] < decay_train[0]
assert decay_norm < free_norm
assert free_acc > 0.8 and decay_acc > 0.8

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(free_train, label="train, no decay")
ax.plot(free_val, label="val, no decay")
ax.plot(decay_val, label="val, decay 0.05")
ax.set_xlabel("step")
ax.set_ylabel("BCE loss")
ax.legend()
ax.set_title("Two moons")
plt.tight_layout()
plt.show()
print("LAB OK")
'''),
        ("md", """
## Your turn

The finite-difference check used `x = 1.5`. Repeat it at `x = -0.3` and confirm the analytic gradient still matches.

## Solution
"""),
        ("code", r'''
z = Value(-0.3)
out = (z * z).tanh()
out.backward()
num = (f(-0.3 + eps) - f(-0.3 - eps)) / (2 * eps)
print(z.grad, num)
assert abs(z.grad - num) < 1e-4
'''),
        ("md", """
## Beyond the course

Train the moons network with SGD at learning rate 0.1 and at learning rate 5, from the same initialization recipe (the seed is reset inside `run_sgd`). The large step size is the ablation. Report the final training loss of each. The stable run should finish lower.
"""),
        ("code", r'''
def run_sgd(lr: float, steps: int = 60) -> float:
    torch.manual_seed(0)
    Xm, ym = make_moons(n_samples=200, noise=0.15, random_state=0)
    X = torch.tensor(Xm, dtype=torch.float32)
    y = torch.tensor(ym, dtype=torch.float32).view(-1, 1)
    model = TorchMLP()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    last = float("nan")
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(X), y)
        loss.backward()
        opt.step()
        last = float(loss.detach())
    return last

low, high = run_sgd(0.1), run_sgd(5.0)
print("lr 0.1", round(low, 3), "lr 5", round(high, 3))
assert low < high
'''),
        ("md", """
## Research questions

1. The scalar engine stores every intermediate `Value`. What does PyTorch store instead, and what does that change about memory during the backward pass?
2. Weight decay lowered the parameter norm. Did validation loss move in the same direction on this dataset? If not, what would you need to see before calling the decay run "better"?
3. `tanh` saturates. Where in the finite-difference check would a gradient near zero be correct rather than a bug?

## Further reading

- Karpathy, micrograd (MIT) and *Neural Networks: Zero to Hero*. The engine above is an original implementation of that design. https://github.com/karpathy/micrograd
- PyTorch tutorials, "Learning PyTorch with Examples" (BSD-style).
- fast.ai Practical Deep Learning, the from-scratch lesson (Apache-2.0 course code). Not copied.
- Goodfellow, Bengio, and Courville, *Deep Learning*, chapter 6, on the chain rule.
"""),
    ]
    emit(
        "labs/training/01_neural_nets_from_scratch/neural_nets_from_scratch.ipynb",
        cells,
        {"track": "training", "lab": "neural_nets", "needs_gpu": False, "needs_api_key": False},
    )


if __name__ == "__main__":
    supervised()
    unsupervised()
    neural()
