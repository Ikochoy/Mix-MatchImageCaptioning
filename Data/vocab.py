#define Vocabulary class
from collections import Counter
import sys
sys.path.insert(0, '../')
from Preproc.AnnotationCleaner import AnnotationCleaner
from transformers import BertTokenizer

class Vocabulary:

    def __init__(self, freq_threshold=100):
        # setting the pre-reserved tokens int to string tokens
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}

        # string to int tokens
        # its reverse dict self.itos
        self.stoi = {v: k for k, v in self.itos.items()}

        self.freq_threshold = freq_threshold
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


    def __len__(self):
        return len(self.itos)

    def tokenize(self,text):
        tokenized_inputs = self.tokenizer(
            text,  # Input text
            add_special_tokens=True,  # add '[CLS]' and '[SEP]'
            padding='max_length',  # pad to a length specified by the max_length
            max_length=280,
            # truncate all sentences longer than max_length -- max # of characters in Twitter
            return_tensors='pt',  # return everything we need as PyTorch tensors
        )
        return tokenized_inputs['input_ids']


    def build_vocab(self, sentence_list):
        frequencies = Counter()
        idx = 4

        #Preproc captions
        cleaned = AnnotationCleaner(sentence_list).clean()

        done = []


        #Tokenize captions + build vocab
        for sentence in cleaned:
            tokenized = self.tokenize(sentence)
            done.append(tokenized)
            for word in tokenized:
                frequencies[word] += 1

                # add the word to the vocab if it reaches minum frequecy threshold
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
        return done

    def numericalize(self, text):
        """ For each word in the text corresponding index token for that word form the vocab built as list """
        tokenized_text = self.tokenize(text)
        return [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
                for token in tokenized_text]



# v = Vocabulary()
# print(v.build_vocab(["hello world"]))
