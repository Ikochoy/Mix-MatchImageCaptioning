import numpy as np

class GloveEmbedder:
    def __init__(self, dim):
        self.file_name = 'glove.6B.{}d.txt'.format(dim)
        self.dim = dim
        self.embeddings = {}

    def load_embeddings(self):
        """load the embeddings from the file with self.dim dimensions
        """
        with open(self.file_name, 'r',  encoding="utf-8") as f:
            full_content = f.read().strip().split('\n')
            for line in full_content:
                splitted_line = line.split(' ')
                word = splitted_line[0]
                embedding = np.array(splitted_line[1:], dtype=np.float32)
                self.embeddings[word] = embedding
        
    
    def get_embedding(self, words_to_idx):
        """Get the embedding for a list of words from a dictionary of words to indices

        Args:
            words_to_idx (dict): a dictonary of words to indices obtained from the Vocabularies class

        Returns:
            np array: a numpy array of shape (len(words_to_idx), self.dim)
        """
        embedding_mtx = np.zeros((len(words_to_idx), self.dim))
        for word, idx in words_to_idx.items():
            if word in self.embeddings:
                embedding_mtx[idx] = self.embeddings[word]
        return embedding_mtx