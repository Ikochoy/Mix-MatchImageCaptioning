# Mix and Match: Image Caption Generators

The generation of image captions has attracted the attention of researchers in the
field of AI due to its many applications in a variety of fields. In this paper, we
explore the impacts of substituting the vision deep CNN component of the caption
generation model presented by Vinyals et al with other pre-trained models. We
experiment with AlexNet, VGG-19, and Inception V3 for the vision component
of the model and the LSTM model from the Vinyals et al paper for the language
generating component of the model and see if there is a difference in performance
in caption generation. We apply these modified models to the Flickr8k dataset
and compare our models’ performances using BLEU-1, BLEU-2, BLEU-3, and
BLEU-4 scores. We find that AlexNet outperforms VGG-19 and Inception V3
when trained with a lower learning rate.
