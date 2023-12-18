import numpy as np
import word_representations

'''

load data; 
for original UD files in conllu format, e.g. 'en_ewt-ud-train.conllu'.
save as Sentence object that contains BERT states, POS tags and dependencies

'''


class Sentence:
    def __init__(self, tokens, pos, states, dependencies, coref, cls_tokens):
        self.tokens = tokens
        self.pos = pos
        self.states = states
        self.dependencies = dependencies
        self.coref = coref
        self.cls_tokens = cls_tokens

    def __str__(self):
        return ' '.join(t for t in self.tokens)

    def labels(self):
        dep_pairs = []
        labels = []
        label_ind = []
        for d in self.dependencies:
            dep_pairs.append(np.concatenate((self.states[d[0]-1], self.states[d[1]-1]), axis=0))
            labels.append(d[2])
        for l in labels:
            label_ind.append(label_to_ix[l])

        return np.stack(dep_pairs), np.array(label_ind, dtype=np.int32)

    def pos_tagging(self):
        return np.stack(self.states[0:len(self.pos)]), np.array(self.pos, dtype=np.int32)

def pos_to_ix(tag):
    tags = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT',
            'SCONJ', 'SYM', 'VERB', 'X', '_']
    tag_to_ix = dict(zip(tags, list(range(len(tags)))))
    tag_index = tag_to_ix[tag]
    return np.array(tag_index, dtype=np.int32)


def label_to_ix(label):
    labels = ['aux', 'nmod', 'cc', 'csubj:pass', 'nmod:npmod', 'acl', 'list', 'root', 'dislocated',
              'case', 'dep', 'csubj:outer', 'vocative', 'acl:relcl', 'discourse', 'mark', 'parataxis',
              'obl', 'xcomp', 'csubj', 'compound', 'nmod:tmod', 'flat:foreign', 'advmod', 'det:predet',
              'punct', 'det', 'ccomp', 'amod', 'obl:tmod', 'advcl:relcl', 'nsubj', 'nsubj:pass', 'aux:pass',
              'nsubj:outer', 'flat', 'cop', 'iobj', 'expl', 'cc:preconj', 'obl:npmod', 'goeswith', 'orphan',
              '_', 'nummod','fixed', 'reparandum', 'conj', 'compound:prt', 'advcl', 'obj', 'appos', 'nmod:poss']
    label_to_ix = dict(zip(labels, list(range(len(labels)))))
    #print(label)
    label_index = label_to_ix[label]
    return np.array(label_index, dtype=np.int32)


def read_conllu(path):
    word_repr = word_representations.Bert()

    with open(path, 'r', encoding='utf-8') as file:
        sentences = []
        pos = []
        deps = []
        coref = []
        tokens = []
        for i, line in enumerate(file):
            if line == '\n':
                text = '[CLS]' + ' '.join(tokens) + '[SEP]'
                pos = [pos_to_ix(tag) for tag in pos]
                states, cls_tokens = word_repr.get_bert(text)
                sentences.append(Sentence(tokens, pos, states, deps, coref, cls_tokens))
                pos = []
                tokens = []
                deps = []
                coref = []
                continue
            if line[0] == '#':
                continue
            line = line.rstrip('\n')
            line = line.split('\t')
            symbols = ['.', ',', '<', '>', ':', ';', '\'', '/', '-', '_', '%', '@', '#', '$', '^', '*', '?', '!', "‘",
                       "’", "'", "+", '=', '|', '\’']
            if len(line[1]) > 1:
                for sym in symbols:
                    line[1] = line[1].replace(sym, '')
            if line[1] == '':
                line[1] = 'unk'
            tokens.append(line[1])
            pos.append(line[3])
            try:
                if int(line[6]) != 0:
                    deps.append((int(line[0]), int(line[6]), label_to_ix(line[7])))

                if line[8].endswith('ref'):
                    ref = line[8].rstrip(':ref')
                    coref.append((int(line[0]), int(ref)))
            except ValueError:
                # print("value error ; the following dependency was not appended:", line[0], line[6], line[7])
                # occurs with index of type '5.1'; rare ; can be ignored
                pass

        return sentences

