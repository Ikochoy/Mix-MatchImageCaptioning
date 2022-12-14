import pandas as pd
import matplotlib.pyplot as plt

# TODO: Update hyperparameters as need be
EMBED_SIZE = 512  # [256, 512]
HIDDEN_SIZE = 256  # [256, 512]
NUM_LAYERS = 1  #
LEARNING_RATE = 3e-2
NUM_EPOCHS = 50  # [25, 50]
BATCH_SIZE = 256  # [128, 256]
NUM_WORKERS = 2

TRAIN_CNN = False

OUTPUT_FOLDER = 'FINAL'

# Pick the model you want to generate the loss curves for
INCEPTION = True
ALEXNET = False
VGG = False

if INCEPTION:
    inception_path = f'{OUTPUT_FOLDER}/encoder=InceptionV3-embed_size={EMBED_SIZE}-hidden_size={HIDDEN_SIZE}-num_layers={NUM_LAYERS}-lr={LEARNING_RATE}-epochs={NUM_EPOCHS}-bs={BATCH_SIZE}-num_workers={NUM_WORKERS}-trainCNN={TRAIN_CNN}/'
    inception_loss_df = pd.read_csv(f'{inception_path}Loss.csv')

    inception_loss_df.rename(columns={'Unnamed: 0': 'Epoch'}, inplace=True)

    inception_loss_df.plot(x="Epoch", y=["train_mean_loss", "val_mean_loss"])
    plt.title("Inception encoder")
    plt.savefig(f'{inception_path}loss-plot.png')
    plt.show()
    # plt.clf()

if ALEXNET:
    alexnet_path = f'{OUTPUT_FOLDER}/encoder=AlexNet-embed_size={EMBED_SIZE}-hidden_size={HIDDEN_SIZE}-num_layers={NUM_LAYERS}-lr={LEARNING_RATE}-epochs={NUM_EPOCHS}-bs={BATCH_SIZE}-num_workers={NUM_WORKERS}-trainCNN={TRAIN_CNN}/'
    alexnet_loss_df = pd.read_csv(f'{alexnet_path}Loss.csv')

    alexnet_loss_df.rename(columns={'Unnamed: 0': 'Epoch'}, inplace=True)

    alexnet_loss_df.plot(x="Epoch", y=["train_mean_loss", "val_mean_loss"])
    plt.title("AlexNet encoder")
    plt.savefig(f'{alexnet_path}loss-plot.png')
    plt.show()
    # plt.clf()

if VGG:
    vgg_path = f'{OUTPUT_FOLDER}/encoder=VGG-embed_size={EMBED_SIZE}-hidden_size={HIDDEN_SIZE}-num_layers={NUM_LAYERS}-lr={LEARNING_RATE}-epochs={NUM_EPOCHS}-bs={BATCH_SIZE}-num_workers={NUM_WORKERS}-trainCNN={TRAIN_CNN}/'
    vgg_loss_df = pd.read_csv(f'{vgg_path}Loss.csv')

    vgg_loss_df.rename(columns={'Unnamed: 0': 'Epoch'}, inplace=True)

    vgg_loss_df.plot(x="Epoch", y=["train_mean_loss", "val_mean_loss"])
    plt.title("VGG encoder")
    plt.savefig(f'{vgg_path}loss-plot.png')
    plt.show()
    # plt.clf()
