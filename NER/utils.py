import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras.layers import LSTM, Embedding, Dense, Dropout, Input, Bidirectional, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import warnings
import os
from sklearn.metrics import accuracy_score


# Ignore specific category warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_prepare_data(file_path):
    logger.info("Loading NER data from file: %s", file_path)
    ner_data = pd.read_csv(file_path, encoding='latin1')
    ner_data.dropna(subset=['Word', 'Tag'], inplace=True)
    ner_data['Sentence #'].fillna(method='ffill', inplace=True)
    grouped_data = ner_data.groupby('Sentence #').apply(lambda s: [(
        w, t) for w, t in zip(s['Word'].values.tolist(), s['Tag'].values.tolist())])
    return [sentence for sentence in grouped_data]


def create_mappings(sentences):
    words = [word for sentence in sentences for word, tag in sentence]
    tags = [tag for sentence in sentences for word, tag in sentence]
    word2idx = {w: i + 1 for i, w in enumerate(set(words))}
    tag2idx = {t: i for i, t in enumerate(set(tags))}
    return word2idx, tag2idx


def process_data(sentences, word2idx, tag2idx, max_len):
    X = [[word2idx.get(w[0], 0) for w in s] for s in sentences]
    X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=0)

    y = [[tag2idx.get(w[1], 0) for w in s] for s in sentences]
    y = pad_sequences(maxlen=max_len, sequences=y,
                      padding="post", value=tag2idx.get("O", 0))
    y = np.array([to_categorical(i, num_classes=len(tag2idx)) for i in y])

    return X, y


def build_lstm_model(word2idx, tag2idx, lstm_units, dense_units, max_len):
    input_layer = Input(shape=(max_len,))
    embedding_layer = Embedding(input_dim=len(
        word2idx) + 1, output_dim=50, input_shape=(max_len,))(input_layer)
    lstm_layer = LSTM(units=lstm_units, return_sequences=True)(embedding_layer)
    dropout_layer = Dropout(0.1)(lstm_layer)
    dense_layer = Dense(dense_units, activation='relu')(dropout_layer)
    output_layer = Dense(len(tag2idx), activation='softmax')(dense_layer)
    model = Model(input_layer, output_layer)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_bilstm_model(word2idx, tag2idx,lstm_units, dense_units, max_len):
    input_layer = Input(shape=(max_len,))
    embedding = Embedding(input_dim=len(word2idx) + 1,
                          output_dim=50, input_shape=(max_len,))(input_layer)
    embedding_dropout = Dropout(0.1)(embedding)

    lstm1 = Bidirectional(
        LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(embedding_dropout)
    lstm1_dropout = Dropout(0.1)(lstm1)
    lstm2 = Bidirectional(
        LSTM(units=50, return_sequences=True, recurrent_dropout=0.1))(lstm1_dropout)
    lstm2_dropout = Dropout(0.1)(lstm2)

    dense1 = TimeDistributed(Dense(50, activation="relu"))(lstm2_dropout)
    dense1_dropout = Dropout(0.1)(dense1)
    dense2 = TimeDistributed(Dense(25, activation="relu"))(dense1_dropout)

    output = TimeDistributed(Dense(len(tag2idx), activation="softmax"))(dense2)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer="adam", loss="categorical_crossentropy",
                  metrics=["accuracy"])

    return model


def train_and_predict_with_model(build_model_func, model_configs, X_train, y_train, X_test, word2idx, tag2idx, max_len, modelname):
    predictions = []
    weights_folder = 'weights'
    for i, (lstm_units, dense_units) in enumerate(model_configs, start=1):
        logger.info(
            f"Training model %d with {modelname} units: %d, Dense units: %d", i, lstm_units, dense_units)
        model = build_model_func(
            word2idx, tag2idx, lstm_units, dense_units, max_len)
        model.fit(X_train, y_train, batch_size=32,
                  epochs=1, validation_split=0.1)
        logger.info("Predicting with model %d", i)
        predictions.append(model.predict(X_test, verbose=1))
        # Create the results folder if it doesn't exist
        if not os.path.exists(weights_folder):
            os.makedirs(weights_folder)
        model.save_weights(os.path.join(weights_folder, f'{modelname}_{i}.weights.h5'))
    final_predictions = np.mean(np.array(predictions), axis=0)
    return final_predictions

def save_results_to_csv(predictions, test_sentences, true_tags, model_name, tag2idx):
    idx2tag = {i: w for w, i in tag2idx.items()}
    pred_labels = np.argmax(predictions, axis=-1)
    pred_tags = [[idx2tag[i] for i in row] for row in pred_labels]

    results = []

    for i, (sentence, true, pred) in enumerate(zip(test_sentences, true_tags, pred_tags)):
        for word, true_tag, pred_tag in zip(sentence, true, pred):
            results.append(
                {"Word": word[0], "True_Tag": true_tag, "Pred_Tag": pred_tag})

    results_df = pd.DataFrame(results)

    # Define the folder to save results
    results_folder = "results"

    # Create the results folder if it doesn't exist
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Define the file path including the folder
    filename = os.path.join(results_folder, f"{model_name}_results.csv")
    logger.info("Results saved to %s", filename)
    results_df.to_csv(filename, index=False)

def get_accuracy_score(predictions, idx2tag, true_labels, model_name):
    pred_labels = np.argmax(predictions, axis=-1)
    pred_tags = [[idx2tag[i] for i in row] for row in pred_labels]
    true_tags = [[idx2tag[i] for i in row] for row in true_labels]
    
    # Flatten the lists
    pred_tags_flat = [tag for sublist in pred_tags for tag in sublist]
    true_tags_flat = [tag for sublist in true_tags for tag in sublist]
    
    # Calculate accuracy
    accuracy = accuracy_score(true_tags_flat, pred_tags_flat)
    logger.info(f'Accuracy for {model_name}: {accuracy}')
    # Evaluation using classification report
    classification_report(true_tags_flat, pred_tags_flat)
    return accuracy
