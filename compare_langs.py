import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
from scipy.spatial.distance import cosine

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
            sentence.append(word)
    return sentences

# Load data
sentences1 = load_conllu_data('en_ewt-ud-train.conllu')
sentences2 = load_conllu_data('fa_perdt-ud-train.conllu')

# Load pre-trained multilingual BERT model
bert_model = hub.load("https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/4")
preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3")

# Encode sentences
def encode_sentences(sentences):
    sentence_embeddings = []
    for sentence in sentences:
        input_data = preprocessor(tf.constant([sentence]))
        outputs = bert_model(input_data)
        sentence_embedding = outputs['pooled_output'][0].numpy()
        sentence_embeddings.append(sentence_embedding)
    return sentence_embeddings

sentence_embeddings1 = encode_sentences(sentences1)
sentence_embeddings2 = encode_sentences(sentences2)

# Compare sentence embeddings
for i, embedding1 in enumerate(sentence_embeddings1):
    for j, embedding2 in enumerate(sentence_embeddings2):
        similarity = 1 - cosine(embedding1, embedding2)
        print(f'Similarity between sentence {i+1} in file 1 and sentence {j+1} in file 2: {similarity:.2f}')
