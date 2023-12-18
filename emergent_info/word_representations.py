import tensorflow as tf
from transformers import BertTokenizer, TFBertModel, BertConfig

class Bert:
    def __init__(self):
        self.config = BertConfig.from_pretrained('TurkuNLP/bert-base-finnish-cased-v1', output_hidden_states=True)
        self.model = TFBertModel.from_pretrained('TurkuNLP/bert-base-finnish-cased-v1', config=self.config)
        self.tokenizer = BertTokenizer.from_pretrained('TurkuNLP/bert-base-finnish-cased-v1', do_basic_tokenize=True)

    @staticmethod
    def collect_pieces(tokenized_text):
        output = []
        curr_token = []
        seq_length = len(tokenized_text)

        for i in range(seq_length):
            curr_piece = tokenized_text[i]
            curr_token.append((i, curr_piece))

            if i < seq_length - 1:
                next_piece = tokenized_text[i + 1]
                if not next_piece.startswith('##'):
                    output.append(curr_token)
                    curr_token = []

        output.append(curr_token)
        return output

    def get_bert(self, text):
        tokenized_text = self.tokenizer.tokenize(text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = tf.constant([indexed_tokens])
        layers = []

        outputs = self.model(tokens_tensor)
        for layer in range(13):
            layers.append(outputs.hidden_states[layer])

        collected_pieces = Bert.collect_pieces(tokenized_text)
        all_states = []
        cls_tokens = []
        for layer in range(13):
            token_states = []
            for t in collected_pieces:
                token_index = t[-1][0]  # taking the last word piece
                token_states.append(layers[layer][0, token_index])

            all_states.append(token_states[1:])
            cls_tokens.append(token_states[0])

        return all_states, cls_tokens
