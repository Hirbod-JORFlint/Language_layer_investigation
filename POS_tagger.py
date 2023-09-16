from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load CoNLL-U data
def load_conllu_data(file_path):
    sentences = []
    sentence = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#'):
                continue
            if not line.strip():
                sentences.append(sentence)
                sentence = []
                continue
            fields = line.strip().split('\t')
            word = fields[1]
            pos = fields[3]
            sentence.append((word, pos))
    return sentences

# Prepare data for training
def prepare_data(sentences, max_len):
    words = []
    tags = []
    for sentence in sentences:
        words.append([word for word, tag in sentence])
        tags.append([tag for word, tag in sentence])
    tokenizer_words = Tokenizer()
    tokenizer_words.fit_on_texts(words)
    tokenizer_tags = Tokenizer()
    tokenizer_tags.fit_on_texts(tags)
    X = tokenizer_words.texts_to_sequences(words)
    X = pad_sequences(X, maxlen=max_len, padding='post')
    y = tokenizer_tags.texts_to_sequences(tags)
    y = pad_sequences(y, maxlen=max_len, padding='post')
    y = np.array([np.eye(len(tokenizer_tags.word_index) + 1)[tag] for tag in y])
    return X, y, tokenizer_words, tokenizer_tags

# Define model
def create_model(vocab_size, max_len):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=max_len))
    model.add(LSTM(units=256, return_sequences=True))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Load and prepare data
sentences = load_conllu_data('fa_perdt-ud-train.conllu')
max_len = max([len(sentence) for sentence in sentences])
X, y, tokenizer_words, tokenizer_tags = prepare_data(sentences, max_len)

# Create and train model
model = create_model(len(tokenizer_words.word_index) + 1, max_len)
model.fit(X, y, batch_size=32, epochs=10)
