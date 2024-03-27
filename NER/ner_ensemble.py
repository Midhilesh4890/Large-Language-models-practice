from utils import *
import logging


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

file_path = 'ner_dataset.csv'
max_len = 50

# Load and prepare data
logger.info("Loading and preparing data")
sentences = load_and_prepare_data(file_path)
logger.info("Data loaded and prepared successfully")

# Split data into train, test, and validation sets
logger.info("Splitting data into train, test, and validation sets")
train_sentences, test_sentences = train_test_split(
    sentences, test_size=0.2, random_state=42)
train_sentences, val_sentences = train_test_split(
    train_sentences, test_size=0.25, random_state=42)
logger.info("Data split completed")

# Create mappings
logger.info("Creating word and tag mappings")
word2idx, tag2idx = create_mappings(train_sentences)
logger.info("Word and tag mappings created successfully")

# Process data
logger.info("Processing data for training and testing")
X_train, y_train = process_data(train_sentences, word2idx, tag2idx, max_len)
X_test, y_test = process_data(test_sentences, word2idx, tag2idx, max_len)
logger.info("Data processed successfully")

# Define the model configurations
model_configs = [(64, 32), (128, 64), (256, 128)]

# Train and predict with LSTM model
logger.info("Training and predicting with LSTM model")
lstm_predictions = train_and_predict_with_model(
    build_lstm_model,
    model_configs,
    X_train,
    y_train,
    X_test,
    word2idx,
    tag2idx,
    max_len,
    'lstm'
)
logger.info("LSTM model trained and predictions made")

# Train and predict with BiLSTM model
logger.info("Training and predicting with BiLSTM model")
bilstm_predictions = train_and_predict_with_model(
    build_bilstm_model,
    model_configs,
    X_train,
    y_train,
    X_test,
    word2idx,
    tag2idx,
    max_len,
    'bilstm'
)
logger.info("BiLSTM model trained and predictions made")

# Convert true labels to tags
true_labels = np.argmax(y_test, axis=-1)
idx2tag = {i: w for w, i in tag2idx.items()}
true_tags = [[idx2tag[i] for i in row] for row in true_labels]

# Save results for LSTM model
logger.info("Saving results for LSTM model")
save_results_to_csv(lstm_predictions, test_sentences,
                    true_tags, "LSTM", tag2idx)
logger.info("Results for LSTM model saved successfully")

# Save results for BiLSTM model
logger.info("Saving results for BiLSTM model")
save_results_to_csv(bilstm_predictions, test_sentences,
                    true_tags, "BiLSTM", tag2idx)
logger.info("Results for BiLSTM model saved successfully")

get_accuracy_score(lstm_predictions, idx2tag, true_labels, "LSTM")
get_accuracy_score(bilstm_predictions, idx2tag, true_labels, "BiLSTM")
# Convert the predictions and true values to label sequences
