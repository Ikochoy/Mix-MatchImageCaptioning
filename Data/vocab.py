#define Vocabulary class
from collections import Counter
import spacy
from Preproc import AnnotationCleaner
from Preproc.EmbedAnnotation import GloveEmbedder


spacy_eng = spacy.load("en")


class Vocabulary:

    def __init__(self, freq_threshold):
        # setting the pre-reserved tokens int to string tokens
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}

        # string to int tokens
        # its reverse dict self.itos
        self.stoi = {v: k for k, v in self.itos.items()}

        self.freq_threshold = freq_threshold
        self.glove = GloveEmbedder(dim=1)
        self.embeddings = None

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenize(text):
        return [token.text.lower() for token in spacy_eng.tokenizer(text)]


    def embedding(self, text):
        return self.glove.get_embedding(text)

    def build_vocab(self, sentence_list):
        frequencies = Counter()
        idx = 4

        #Preproc captions
        cleaned = AnnotationCleaner(sentence_list)


        #Tokenize captions + build vocab
        for sentence in cleaned:
            tokenized = self.tokenize(sentence)
            for word in tokenized:
                frequencies[word] += 1

                # add the word to the vocab if it reaches minum frequecy threshold
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

        #returns word embeddings
        self.embeddings = self.embedding(self.stoi)
        return self.embeddings

    def numericalize(self, text):
        """ For each word in the text corresponding index token for that word form the vocab built as list """
        tokenized_text = self.tokenize(text)
        return [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
                for token in tokenized_text]
