import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import threading
import time
import psutil
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
sentences_esl = load_conllu_data('/content/drive/MyDrive/Train_dat/en_esl-ud-train.conllu')
sentences_eslspok = load_conllu_data('/content/drive/MyDrive/Train_dat/en_eslspok-ud-train.conllu')
sentences_en_gum = load_conllu_data('/content/drive/MyDrive/Train_dat/en_gum-ud-train.conllu')
sentences_en_partut = load_conllu_data('/content/drive/MyDrive/Train_dat/en_partut-ud-train.conllu')
sentences_en_ewt = load_conllu_data('/content/drive/MyDrive/Train_dat/en_ewt-ud-train.conllu')
sentences_fa_seraj = load_conllu_data('/content/drive/MyDrive/Train_dat/fa_seraji-ud-train.conllu')
sentences_fa_perdt = load_conllu_data('/content/drive/MyDrive/Train_dat/fa_perdt-ud-train.conllu')

# Load pre-trained multilingual BERT model
bert_model = hub.load("https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/4")
preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3")

# Encode sentences
def encode_sentences(sentences, batch_size=64):
    sentence_embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i+batch_size]
        sentence_strs = [' '.join(sentence) for sentence in batch_sentences]
        input_data = preprocessor(sentence_strs)
        outputs = bert_model(input_data)
        batch_embeddings = outputs['pooled_output'].numpy()
        sentence_embeddings.extend(batch_embeddings)
    return sentence_embeddings

# Encode all previous sentences with batch processing
sentence_embeddings_esl = encode_sentences(sentences_esl, batch_size=128)
sentence_embeddings_eslspok = encode_sentences(sentences_eslspok, batch_size=128)
sentence_embeddings_en_gum = encode_sentences(sentences_en_gum, batch_size=128)
sentence_embeddings_en_partut = encode_sentences(sentences_en_partut, batch_size=128)
sentence_embeddings_en_ewt = encode_sentences(sentences_en_ewt, batch_size=128)
sentence_embeddings_fa_seraj = encode_sentences(sentences_fa_seraj, batch_size=128)
sentence_embeddings_fa_perdt = encode_sentences(sentences_fa_perdt, batch_size=128)


# Define a function to compare and write similarities to a file
considering the following function and operation 33 rewrite my compare function to maximize performance

def compare_and_write_similarities(file1_name, sentences1, file2_name, sentences2, output_file):
    with open(output_file, 'w') as f:
        #f.write(f"Comparison between {file1_name} and {file2_name}:\n")
        #f.write("Sentence1, Sentence2, Similarity\n")
        first_line="first_line_no,second_line_no,Similarity"
        f.write(first_line)
        for i, embedding1 in enumerate(sentences1):
            for j, embedding2 in enumerate(sentences2):
                similarity = 1 - cosine(embedding1, embedding2)
                result = "line{},line{},{:.4f}".format(i+1,j+1,similarity)
                result+="\n"
                f.write(result)
#operation 33
datasets1 = [
    ("en_esl", sentence_embeddings_esl),
    ("en_eslspok", sentence_embeddings_eslspok),
    ("en_gum", sentence_embeddings_en_gum),
    ("en_partut", sentence_embeddings_en_partut),
    ("en_ewt", sentence_embeddings_en_ewt)
]
datasets2 = [
    ("fa_seraj", sentence_embeddings_fa_seraj),
    ("fa_perdt", sentence_embeddings_fa_perdt),
]
for file1_name,sentences1 in datasets1:
  for file2_name,sentences2 in datasets2:
    output_file = f"{file1_name}_{file2_name}_comparison.txt"
    compare_and_write_similarities(file1_name, sentences1, file2_name, sentences2, output_file)
