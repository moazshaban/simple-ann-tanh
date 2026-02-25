import numpy as np

# -------- Activation Function --------
def tanh(x):
    return np.tanh(x)

# -------- Inputs --------
i1 = 0.05
i2 = 0.10

X = np.array([i1, i2])

# -------- Random Weights [-0.5 , 0.5] --------

# Input -> Hidden (2x2)
W1 = np.random.uniform(-0.5, 0.5, (2,2))

# Hidden -> Output (2x2)
W2 = np.random.uniform(-0.5, 0.5, (2,2))

# -------- Biases --------

b1 = 0.5   # Hidden layer bias
b2 = 0.7   # Output layer bias

# -------- Forward Pass --------

# Hidden Layer
net_h = np.dot(X, W1) + b1
out_h = tanh(net_h)

# Output Layer
net_o = np.dot(out_h, W2) + b2
out_o = tanh(net_o)

# -------- Results --------

print("Input values:")
print(X)

print("\nWeights Input -> Hidden:")
print(W1)

print("\nWeights Hidden -> Output:")
print(W2)

print("\nHidden Layer Output:")
print(out_h)

print("\nFinal Network Output:")
print(out_o)