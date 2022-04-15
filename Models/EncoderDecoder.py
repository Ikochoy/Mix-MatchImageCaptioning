import torch.nn as nn
# import image encoder 
from .ImageEncoder import ImageEncoder
# import caption decoder
from .SentenceDecoder import SentenceDecoder

class EncoderDecoder(nn.Module):
    '''
    Encode and decode a image and caption pair.
    '''

    def __init__(
        self, encoder_choice, decoder_choice, vocab_size, encoder_dropout=False, 
        hidden_size=512, word_embedding_size=1000
    ):
        # ToDo: fix word_embedding_size
        super(EncoderDecoder, self).__init__()
        self.encoder_choice = encoder_choice
        self.decoder_choice = decoder_choice
        self.vocab_size = vocab_size
        self.encoder_dropout = encoder_dropout
        self.hidden_size = hidden_size
        self.word_embedding_size = word_embedding_size
        self.image_encoder = ImageEncoder(encoder_choice, encoder_dropout)
        self.sentence_decoder = SentenceDecoder(decoder_choice, vocab_size, hidden_size)        

    def forward(self, images, captions):
        '''
        images: batch_size x 3 x 224 x 224
        captions: batch_size x max_caption_length
        '''
        # encode image
        image_features = self.image_encoder(images)
        # encode caption
        return self.sentence_decoder.teacher_forcing(image_features, captions)





