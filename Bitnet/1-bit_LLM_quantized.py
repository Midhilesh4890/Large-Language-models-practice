import numpy as np


def round_clip(x, min_val, max_val):
    return max(min(max_val, round(x)), min_val)


def absmean_quantization(W):
    # Calculate scaling factor σ
    sigma = 1 / (1/np.size(W) * np.sum(np.abs(W)))

    # Scale the weight matrix by σ
    scaled_W = W * sigma

    # Apply RoundClip function to each element of the scaled matrix
    quantized_W = np.vectorize(lambda x: round_clip(x, -1, 1))(scaled_W)

    return quantized_W


def absmean_quantization_activation(A):
    # Find the maximum absolute value in the activation array
    max_val = np.max(np.abs(A))

    # Avoid scaling factor that would make zero a quantized value
    scaling_factor = 1 / max_val if max_val != 0 else 1

    # Scale the activation matrix to the range [-1, 1]
    scaled_A = A * scaling_factor

    # Apply RoundClip function to each element of the scaled matrix
    quantized_A = np.vectorize(lambda x: round_clip(x, -1, 1))(scaled_A)

    return quantized_A


# Example weight matrix
W = np.array([[-0.5, 0.8, -1.2],
              [1.0, -0.3, 0.6],
              [0.4, -0.9, 1.5]])

# Example activation matrix
A = np.array([-0.7, 0.9, -1.3, 1.1, -0.4])

# Apply absmean quantization to weights
quantized_weights = absmean_quantization(W)
print("Quantized Weights:")
print(quantized_weights)

# Apply absmean quantization to activations
quantized_activations = absmean_quantization_activation(A)
print("Quantized Activations:")
print(quantized_activations)

# Link to access
# https://arxiv.org/html/2402.17764v1
