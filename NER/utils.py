import pandas as pd
import numpy as np
import random
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras.layers import LSTM, Embedding, Dense, Dropout, Input, Bidirectional, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import warnings

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
    logger.info("Creating word and tag mappings")
    words = [word for sentence in sentences for word, tag in sentence]
    tags = [tag for sentence in sentences for word, tag in sentence]
    word2idx = {w: i + 1 for i, w in enumerate(set(words))}
    tag2idx = {t: i for i, t in enumerate(set(tags))}
    return word2idx, tag2idx


def process_data(sentences, word2idx, tag2idx, max_len):
    logger.info("Processing data for training")
    X = [[word2idx.get(w[0], 0) for w in s] for s in sentences]
    X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=0)

    y = [[tag2idx.get(w[1], 0) for w in s] for s in sentences]
    y = pad_sequences(maxlen=max_len, sequences=y,
                      padding="post", value=tag2idx.get("O", 0))
    y = np.array([to_categorical(i, num_classes=len(tag2idx)) for i in y])

    return X, y


def build_lstm_model(word2idx, tag2idx, lstm_units, dense_units, max_len):
    logger.info("Building LSTM model")
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


def build_bilstm_model(word2idx, tag2idx, lstm_units, dense_units, max_len):
    logger.info("Building BiLSTM model")
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


def train_and_predict_with_model(build_model_func, model_configs, X_train, y_train, X_test, word2idx, tag2idx, max_len):
    predictions = []
    for i, (lstm_units, dense_units) in enumerate(model_configs, start=1):
        logger.info(
            "Training model %d with LSTM units: %d, Dense units: %d", i, lstm_units, dense_units)
        model = build_model_func(
            word2idx, tag2idx, lstm_units, dense_units, max_len)
        model.fit(X_train, y_train, batch_size=32,
                  epochs=1, validation_split=0.1)
        logger.info("Predicting with model %d", i)
        predictions.append(model.predict(X_test, verbose=1))
        # Optionally save the model weights
        # model.save_weights(f'ner_model_{i}.weights.h5')
    return np.mean(np.array(predictions), axis=0)

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

    filename = f"{model_name}_results.csv"
    results_df.to_csv(filename, index=False)

    logger.info("Results saved to %s", filename)
