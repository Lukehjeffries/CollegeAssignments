

def relu(x):
    return x if x > 0 else 0

def forward_four_neurons(weights, biases, inputs):
    assert len(weights) == 4 and len(biases) == 4 and len(inputs) == 3
    outputs = []
    for n in range(4):
        print(f"\n=== Neuron {n+1} ===")
        z = 0
        for i in range(3):
            prod = weights[n][i] * inputs[i]
            print(f"w[{n+1}][{i+1}] * x[{i+1}] = {weights[n][i]} * {inputs[i]} = {prod}")
            z += prod
            print(f"  partial sum = {z}")
        print(f"adding bias b[{n+1}] = {biases[n]}")
        z += biases[n]
        print(f"z_{n+1} = {z}")
        y = relu(z)
        print(f"ReLU -> output y_{n+1} = {y}")
        outputs.append(y)
    return outputs

if __name__ == "__main__":

    inputs = [1, 1, 1]
    w1 = [2, -1, 3]; b1 = -5
    w2 = [2, 0, 1]; b2 = 0
    w3 = [1, 3, 1]; b3 = 0
    w4 = [2, 3, -2]; b4 = 0

    weights = [w1, w2, w3, w4]
    biases = [b1, b2, b3, b4]

    outputs = forward_four_neurons(weights, biases, inputs)
    print("\nLayer outputs (after ReLU):", outputs)

    # Parameter counts
    per_neuron_params = len(inputs) + 1 
    total_params = len(weights) * per_neuron_params
    print(f"Parameters per neuron: {per_neuron_params} (3 weights + 1 bias)")
    print(f"Total parameters in layer: {total_params}")
