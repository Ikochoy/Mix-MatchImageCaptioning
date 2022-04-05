from ..Data.vocab import Vocabulary

# Build list of vocabs
threshold = 4
def build_giant_list_of_vocabs(imageset):
  words = {}
  words_count = {}
  vocabs = Vocabulary()
  sentence_list = []
  for img_id, image in imageset.images.items():
    for caption in image.annotations:
      for word in caption.split():
        if word not in words_count:
          words_count[word] = 1
        else:
          words_count[word] += 1
        if words_count[word] > threshold:
          words.add(word)
          sentence_list.append(word)

  vocabs.build_vocab(sentence_list)
  # return a set of words, so that there would not be duplicates
  return words, words_count
