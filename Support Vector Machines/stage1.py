import numpy as np


def hard_margin_objective(w):

    return 0.5 * np.dot(w, w)


def satisfies_hard_margin_constraints(w, b, X, y):
    margins = y * (X @ w + b)
    return np.all(margins >= 1)


def hinge_loss_values(w, b, X, y):
    margins = y * (X @ w + b)
    return np.maximum(0, 1 - margins)


def soft_margin_objective(w, b, X, y, C):
    loss = hinge_loss_values(w, b, X, y)
    return 0.5 * np.dot(w, w) + C * np.sum(loss)

