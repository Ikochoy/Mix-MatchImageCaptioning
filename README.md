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

To train the models: 
1. Clone this repository  
2. Modify the following at the top of train.py to your preferences: \
  `ENCODER_CHOICE`: Choice of InceptionV3, AlexNet, and VGG\
  `LOAD_MODEL`: set to True if you want to load a model that has already been partially trained\
  `TRAIN_CNN`: True or False. If False, only fine-tune. If True, train the full CNN\
  `DEVICE`: GPU or CPU\
   hyperparameters: `EMBED_SIZE`. `HIDDEN_SIZE`, `NUM_LAYERS`, `LEARNING_RATE`, `NUM_EPOCHS`, `BATCH_SIZE`, `NUM_WORKERS`
3. Run train.py 

Read our project report [here](ProjectReport.pdf). 
