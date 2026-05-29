
import numpy as np

def relu(x):
    return np.maximum(0, x)


W = np.array([
    [2, -1, 3],
    [0, 1, -2],
    [1, 3, 1],
    [2, -1, 0]
])


b = np.array([[-1], [1], [0], [2]])


X = np.array([
    [1, 0, 2],
    [2, 3, 1],
    [-1, 1, 0]
])

Z = W @ X + b
H = relu(Z)

print("Weighted sums (Z):")
print(Z)
print("\nReLU activations (H):")
print(H)
