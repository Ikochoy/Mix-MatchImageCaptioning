class Vocabularies:
  def __init__(self):
    self.word_to_idx = {}
    self.idx_to_word = {}
    self.len = 0

  def add_word(self, word):
    if word not in self.word_to_idx:
      self.word_to_idx[word] = self.len + 1
      self.len += 1
      self.idx_to_word[self.len] = word