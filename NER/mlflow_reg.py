from utils import *
import logging
import mlflow
from mlflow.models.signature import infer_signature

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

file_path = 'ner_dataset.csv'
max_len = 50

# Initialize MLflow experiment
mlflow.set_experiment("NER_Model_Training")
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

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

# Convert true labels to tags
true_labels = np.argmax(y_test, axis=-1)
idx2tag = {i: w for w, i in tag2idx.items()}
true_tags = [[idx2tag[i] for i in row] for row in true_labels]

# Define the model configurations
model_configs = [(64, 32), (128, 64), (256, 128)]

for config in model_configs:
    lstm_units, dense_units = config

    # Start a new MLflow run for each model configuration
    with mlflow.start_run(run_name="LSTM_{}_{}".format(lstm_units, dense_units)):
        # Log parameters
        mlflow.log_param("lstm_units", lstm_units)
        mlflow.log_param("dense_units", dense_units)
        mlflow.log_param("file_path", file_path)
        mlflow.log_param("max_len", max_len)

        # Train and predict with LSTM model
        logger.info("Training and predicting with LSTM model")
        lstm_predictions = train_and_predict_with_model(
            build_lstm_model,
            [(lstm_units, dense_units)],
            X_train,
            y_train,
            X_test,
            word2idx,
            tag2idx,
            max_len,
            'lstm'
        )
        logger.info("LSTM model trained and predictions made")
        # Save the model
        mlflow.keras.log_model(build_lstm_model(
            word2idx, tag2idx, lstm_units, dense_units, max_len), "model")
        # Log metrics (example: accuracy)
        calculated_accuracy = get_accuracy_score(lstm_predictions, idx2tag, true_labels, "LSTM")
        mlflow.log_metric("accuracy", calculated_accuracy)
       
logger.info("Experiment completed. Results logged to MLflow.")
