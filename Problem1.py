

def relu(x):
    return max(0, x)

w = [1, -1, 1]
x = [2, 1, 3]
b = -5

# Weighted sum: w*x + b
weighted_sum = sum(w_i * x_i for w_i, x_i in zip(w, x)) + b

output = relu(weighted_sum)

print("Weighted sum (w·x + b):", weighted_sum)
print("ReLU output:", output)
