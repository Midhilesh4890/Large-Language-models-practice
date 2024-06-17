import numpy as np
import math
import logging

# Configure logging
logging.basicConfig(filename='transformer_steps.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Starting Transformer explanation script.")

# Step 1: Defining our Dataset
dataset = [
    "when you play the game of thrones",
    "you win or you die",
    "there is no middle ground"
]
logging.info("Dataset defined: %s", dataset)

# Step 2: Finding Vocab Size
words = " ".join(dataset).split()
unique_words = set(words)
vocab_size = len(unique_words)
logging.info("Vocabulary Size: %d", vocab_size)
logging.info("Unique Words: %s", unique_words)

# Step 3: Encoding
word_to_index = {word: idx for idx, word in enumerate(unique_words)}
logging.info("Word to Index Mapping: %s", word_to_index)
encoded_dataset = [[word_to_index[word] for word in sentence.split()] for sentence in dataset]
logging.info("Encoded Dataset: %s", encoded_dataset)

# Step 4: Calculating Embedding
embedding_dim = 6
embedding_matrix = np.random.rand(vocab_size, embedding_dim)
logging.info("Embedding Matrix: %s", embedding_matrix)
sentence = "when you play the game of thrones"
encoded_sentence = [word_to_index[word] for word in sentence.split()]
embedded_sentence = np.array([embedding_matrix[idx] for idx in encoded_sentence])
logging.info("Embedded Sentence: %s", embedded_sentence)

# Step 5: Calculating Positional Embedding
def positional_encoding(position, d_model):
    angle_rads = np.array([pos / np.power(10000, 2 * (i // 2) / d_model) for pos in range(position) for i in range(d_model)])
    angle_rads[0::2] = np.sin(angle_rads[0::2])
    angle_rads[1::2] = np.cos(angle_rads[1::2])
    return angle_rads.reshape(position, d_model)

positional_embedding = positional_encoding(len(sentence.split()), embedding_dim)
logging.info("Positional Embedding: %s", positional_embedding)

# Step 6: Concatenating Positional and Word Embeddings
concatenated_embedding = embedded_sentence + positional_embedding
logging.info("Concatenated Embedding: %s", concatenated_embedding)

# Step 7: Multi Head Attention
def scaled_dot_product_attention(query, key, value, mask=None):
    matmul_qk = np.dot(query, key.T)
    dk = key.shape[-1]
    scaled_attention_logits = matmul_qk / np.sqrt(dk)

    # Improve numerical stability by subtracting the max value
    max_logits = np.max(scaled_attention_logits, axis=-1, keepdims=True)
    scaled_attention_logits -= max_logits

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    exp_logits = np.exp(scaled_attention_logits)
    sum_exp_logits = np.sum(exp_logits, axis=-1, keepdims=True)
    attention_weights = exp_logits / (sum_exp_logits + 1e-9)  # Adding a small constant to prevent division by zero

    output = np.dot(attention_weights, value)
    return output, attention_weights

query = concatenated_embedding
key = concatenated_embedding
value = concatenated_embedding

attention_output, attention_weights = scaled_dot_product_attention(query, key, value)
logging.info("Attention Output: %s", attention_output)
logging.info("Attention Weights: %s", attention_weights)

# Step 8: Adding and Normalizing
def layer_norm(x, epsilon=1e-6):
    mean = np.mean(x, axis=-1, keepdims=True)
    std_dev = np.std(x, axis=-1, keepdims=True)
    normalized = (x - mean) / (std_dev + epsilon)
    return normalized

add_norm_output = layer_norm(attention_output + concatenated_embedding)
logging.info("Add and Norm Output: %s", add_norm_output)

# Step 9: Feed Forward Network
def feed_forward_network(x, hidden_dim):
    weights1 = np.random.rand(x.shape[-1], hidden_dim)
    bias1 = np.random.rand(hidden_dim)
    weights2 = np.random.rand(hidden_dim, x.shape[-1])
    bias2 = np.random.rand(x.shape[-1])
    
    output = np.dot(x, weights1) + bias1
    output = np.maximum(0, output)  # ReLU
    output = np.dot(output, weights2) + bias2
    return output

ffn_output = feed_forward_network(add_norm_output, hidden_dim=4)
logging.info("Feed Forward Network Output: %s", ffn_output)

# Step 10: Adding and Normalizing Again
add_norm_ffn_output = layer_norm(ffn_output + add_norm_output)
logging.info("Add and Norm After FFN Output: %s", add_norm_ffn_output)

# Step 11: Decoder Part (simplified as described)
def masked_multi_head_attention(query, key, value, mask):
    attn_output, attn_weights = scaled_dot_product_attention(query, key, value, mask)
    return attn_output

mask = np.tril(np.ones((len(sentence.split()), len(sentence.split()))))

decoder_query = add_norm_ffn_output
decoder_key = concatenated_embedding
decoder_value = concatenated_embedding

decoder_attention_output = masked_multi_head_attention(decoder_query, decoder_key, decoder_value, mask)
logging.info("Decoder Attention Output: %s", decoder_attention_output)

# Step 12: Understanding Masked Multi-Head Attention
input_matrix = np.random.rand(4, 6)
logging.info("Input Matrix for Masked Attention: %s", input_matrix)

head_1_weights = np.eye(6)
head_2_weights = np.eye(6)
mask = np.array([
    [0, -np.inf, -np.inf, -np.inf],
    [0, 0, -np.inf, -np.inf],
    [0, 0, 0, -np.inf],
    [0, 0, 0, 0]
])

head_1_output = np.dot(input_matrix, head_1_weights)
head_2_output = np.dot(input_matrix, head_2_weights)

combined_output = np.concatenate((head_1_output, head_2_output), axis=-1)
logging.info("Combined Head Output: %s", combined_output)

# Step 13: Calculating the Predicted Word
logits = np.random.rand(vocab_size)
softmax_logits = np.exp(logits) / (np.sum(np.exp(logits)) + 1e-9)
predicted_word_idx = np.argmax(softmax_logits)
predicted_word = list(unique_words)[predicted_word_idx]
logging.info("Predicted Word: %s", predicted_word)

# Conclusion
logging.info("Transformer process explained with code and calculations.")
