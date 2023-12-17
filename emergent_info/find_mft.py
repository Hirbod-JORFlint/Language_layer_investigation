from collections import defaultdict, Counter
from utils import get_tensors
import torch


# dict word : word-specific most frequent tag
class TagdictTagger:

    def __init__(self):
        self.tags = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT',
            'SCONJ', 'SYM', 'VERB', 'X']
        self.tagdict = {}
        self.most_frequent_tag = None

    def train(self, data):
        tagdict = defaultdict(Counter)
        counter = Counter()
        for tagged_sentence in data:
            for word, tag in zip(tagged_sentence.tokens, tagged_sentence.pos):
                tagdict[word][tag] += 1
                counter[tag] += 1
        for word in tagdict:
            self.tagdict[word] = tagdict[word].most_common(1)[0][0]
        self.most_frequent_tag = counter.most_common(1)[0][0]

    def tag(self, word):
        return self.tagdict[word] if word in self.tagdict else self.most_frequent_tag


def make_sets(sentences, sentences_val, layer1, layer2):
    Td = TagdictTagger()
    Td.train(sentences)

    X_easy = []
    y_easy = []
    X_normal = []
    y_normal = []

    for s in sentences:
        if len(s.pos) == len(s.states[layer1][:-1]):
            for i in range(len(s.pos) - 1):
                c = torch.cat((s.states[layer1][i], s.states[layer2][i]),0)
                X_normal.append(c)
                y_normal.append(s.pos[i])
                if s.pos[i] == Td.tag(s.tokens[i]):
                    X_easy.append(c)
                    y_easy.append(s.pos[i])

    X_val_hard = []
    y_val_hard = []
    X_val_easy = []
    y_val_easy = []
    X_val_normal = []
    y_val_normal = []

    for s in sentences_val:
        if len(s.pos) == len(s.states[layer1][:-1]):
            for i in range(len(s.pos) - 1):
                c = torch.cat((s.states[layer1][i], s.states[layer2][i]),0)
                X_val_normal.append(c)
                y_val_normal.append(s.pos[i])
                if s.pos[i] != Td.tag(s.tokens[i]):
                    X_val_hard.append(c)
                    y_val_hard.append(s.pos[i])
                else: 
                    X_val_easy.append(c)
                    y_val_easy.append(s.pos[i])

    t_X_easy, t_y_easy = get_tensors(X_easy, y_easy)
    t_X_normal, t_y_normal = get_tensors(X_normal, y_normal)
    t_X_val_hard, t_y_val_hard = get_tensors(X_val_hard, y_val_hard)
    t_X_val_easy, t_y_val_easy = get_tensors(X_val_easy, y_val_easy)
    t_X_val_normal, t_y_val_normal = get_tensors(X_val_normal, y_val_normal)
    return t_X_normal, t_y_normal, t_X_easy, t_y_easy, t_X_val_normal, t_y_val_normal, t_X_val_hard, t_y_val_hard, t_X_val_easy, t_y_val_easy
