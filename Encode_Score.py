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
def encode_sentences(sentences):
    sentence_embeddings = []
    for sentence in sentences:
        sentence_str = ' '.join(sentence)
        input_data = preprocessor(tf.constant([sentence_str]))
        outputs = bert_model(input_data)
        sentence_embedding = outputs['pooled_output'][0].numpy()
        sentence_embeddings.append(sentence_embedding)
    return sentence_embeddings

sentence_embeddings_esl = encode_sentences(sentences_esl)
sentence_embeddings_eslspok = encode_sentences(sentences_eslspok)
sentence_embeddings_en_gum = encode_sentences(sentences_en_gum)
sentence_embeddings_en_partut = encode_sentences(sentences_en_partut)
sentence_embeddings_en_ewt = encode_sentences(sentences_en_ewt)
sentence_embeddings_fa_seraj = encode_sentences(sentences_fa_seraj)
sentence_embeddings_fa_perdt = encode_sentences(sentences_fa_perdt)

# Define a function to compare and write similarities to a file
class CompareAndWriteSimilaritiesThread(threading.Thread):
    def __init__(self, file1_name, sentences1, file2_name, sentences2, output_file, lock):
        super().__init__()
        self.file1_name = file1_name
        self.sentences1 = sentences1
        self.file2_name = file2_name
        self.sentences2 = sentences2
        self.output_file = output_file
        self.lock = lock

    def run(self):
        with open(self.output_file, 'a') as f:
            for i, embedding1 in enumerate(self.sentences1):
                for j, embedding2 in enumerate(self.sentences2):
                    similarity = 1 - cosine(embedding1, embedding2)
                    result = "line{},line{},{:.4f}".format(i+1,j+1,similarity)
                    self.lock.acquire()
                    f.write(result)
                    self.lock.release()

def compare_and_write_similarities_parallel(file1_name, sentences1, file2_name, sentences2, output_file):
    lock = threading.Lock()
    threads = []
    for i, embedding1 in enumerate(sentences1):
        for j, embedding2 in enumerate(sentences2):
            thread = CompareAndWriteSimilaritiesThread(file1_name, sentences1, file2_name, sentences2, output_file, lock)
            threads.append(thread)
            thread.start()

    for thread in threads:
        thread.join()

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

# Take system memory into account
available_memory = psutil.virtual_memory().available
# Limit the number of threads to avoid overloading the system
max_threads = int(available_memory / (1024 * 1024))
if max_threads < 1:
    max_threads = 1

for file1_name,sentences1 in datasets1:
  for file2_name,sentences2 in datasets2:
    output_file = f"{file1_name}_{file2_name}_comparison.txt"
    compare_and_write_similarities_parallel(file1_name, sentences1, file2_name, sentences2, output_file)
