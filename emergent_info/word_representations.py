import torch
from transformers import BertTokenizer, BertModel, BertConfig

"""

class Bert

access via get_bert()
input: sentence represented as a string
returns: list with torch states 

"""

# All models: 
# 'bert-base-uncased', 'bert-base-german-cased', 'TurkuNLP/bert-base-finnish-cased-v1', 'onlplab/alephbert-base' (he), 'KB/bert-base-swedish-cased', 
# 'UWB-AIR/Czert-B-base-cased', dbmdz/bert-base-turkish-cased

class Bert:
    def __init__(self):
        self.config = BertConfig.from_pretrained('TurkuNLP/bert-base-finnish-cased-v1' , output_hidden_states=True)
        self.model = BertModel.from_pretrained('TurkuNLP/bert-base-finnish-cased-v1' , config=self.config)
        self.tokenizer = BertTokenizer.from_pretrained('TurkuNLP/bert-base-finnish-cased-v1' , do_basic_tokenize=True)
        self.model.eval()

    # align word pieces with words
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
        tokens_tensor = torch.tensor([indexed_tokens])
        layers = []

        with torch.no_grad():
            outputs = self.model(tokens_tensor)
            for layer in range(13):
                #print(layer)
                layers.append(outputs[2][layer])

        collected_pieces = Bert.collect_pieces(tokenized_text)
        all_states = []
        cls_tokens = []
        for layer in range(0,13):
            token_states = []
            for t in collected_pieces:
                token_index = t[-1][0]  # taking last word piece
                token_states.append(layers[layer][0, token_index])
                token_states_new = token_states[1:len(token_states)]

            all_states.append(token_states_new)
            cls_tokens.append(token_states[0])
            
        return all_states, cls_tokens

