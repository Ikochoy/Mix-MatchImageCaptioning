from .Vocabularies import Vocabularies

# Build list of vocabs 
threshold = 4
def build_giant_list_of_vocabs(imageset):
  words = {}
  words_count = {}
  vocabs = Vocabularies()
  for img_id, image in imageset.images.items():
    for caption in image.annotations:
      for word in caption.split():
        if word not in words_count:
          words_count[word] = 1
        else:
          words_count[word] += 1
        if words_count[word] > threshold:
          words.add(word)
          vocabs.add_word(word)
  # return a set of words, so that there would not be duplicates
  return words, words_count