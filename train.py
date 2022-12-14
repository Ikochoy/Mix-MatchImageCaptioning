import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint, load_checkpoint, print_examples
from get_loader import get_loader
from model import EncoderToDecoder

import numpy as np
import pandas as pd
from pathlib import Path

# Clear GPU cache
torch.cuda.empty_cache()

# Encoder input sizes so we can appropriately resize the images as per the encoder choice
encoder_input_sizes = {'InceptionV3': 299, 'AlexNet': 227, 'VGG': 244}  # TODO: Check if right for AlexNet and VGG

# TODO: CHOICES
ENCODER_CHOICE = 'InceptionV3'  # ['InceptionV3', 'AlexNet', 'VGG']

LOAD_MODEL = False
LOAD_MODEL_PATH = "my_checkpoint.pth.tar"

SAVE_MODEL = True
TRAIN_CNN = False  # If False, only fine-tune. If True, train the full CNN

# TODO: Update hyperparameters as need be
EMBED_SIZE = 512  # [256, 512]
HIDDEN_SIZE = 256  # [256, 512]
NUM_LAYERS = 1  #
LEARNING_RATE = 3e-4
NUM_EPOCHS = 50  # [25, 50]
BATCH_SIZE = 256  # [128, 256]
NUM_WORKERS = 1

# Set DEVICE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# Determines the OUTPUT FOLDER and where to store results
OUTPUT_FOLDER = f"FINAL/encoder={ENCODER_CHOICE}-embed_size={EMBED_SIZE}-hidden_size={HIDDEN_SIZE}-num_layers={NUM_LAYERS}-lr={LEARNING_RATE}-epochs={NUM_EPOCHS}-bs={BATCH_SIZE}-num_workers={NUM_WORKERS}-trainCNN={TRAIN_CNN}/"
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)


def train():
    # ------------------------------------------------------------------------------------------------------------------
    # Image transforms to apply to dataset images
    transform = transforms.Compose(
        [
            transforms.Resize((encoder_input_sizes[ENCODER_CHOICE], encoder_input_sizes[ENCODER_CHOICE])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Taken from ImageNet
        ]
    )

    # Get the loaders and datasets for train-val-test
    train_loader, train_dataset = get_loader(
        root_folder="flickr8k/images",
        annotation_file="flickr8k/train_captions.txt",
        transform=transform,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    print(f"Vocab size = {len(train_dataset.vocab)}")
    val_loader, val_dataset = get_loader(
        root_folder="flickr8k/images",
        annotation_file="flickr8k/val_captions.txt",
        transform=transform,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    test_loader, test_dataset = get_loader(
        root_folder="flickr8k/images",
        annotation_file="flickr8k/test_captions.txt",
        transform=transform,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    # ------------------------------------------------------------------------------------------------------------------

    torch.backends.cudnn.benchmark = True

    # Get vocab size
    VOCAB_SIZE = len(train_dataset.vocab)

    # For TensorBoard
    writer = SummaryWriter("runs/flickr")
    step = 0

    # Initialize model, loss and optimizer
    model = EncoderToDecoder(ENCODER_CHOICE, EMBED_SIZE, HIDDEN_SIZE, VOCAB_SIZE, NUM_LAYERS).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # THIS WAS JUST FOR DUMMY SAKE
    # DUMMY_model = EncoderToDecoder(ENCODER_CHOICE, EMBED_SIZE, HIDDEN_SIZE, VOCAB_SIZE, NUM_LAYERS).to(DEVICE)
    # DUMMY_optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Only finetune the CNN if TRAIN_CNN is False.
    # Train the full CNN is TRAIN_CNN is True
    for name, param in model.encoderCNN.model.named_parameters():

        # Determines name of layer we want to tune in respective encoder architectures
        if ENCODER_CHOICE == 'InceptionV3':
            weight = "fc.weight"
            bias = "fc.bias"
        elif ENCODER_CHOICE == "AlexNet" or ENCODER_CHOICE == "VGG":
            weight = "classifier.6.weight"
            bias = "classifier.6.bias"

        if weight in name or bias in name:
            param.requires_grad = True
        else:
            param.requires_grad = TRAIN_CNN

    # TODO: CHECK WHERE YOU ARE LOADING MODELS FROM
    if LOAD_MODEL:
        step = load_checkpoint(torch.load(LOAD_MODEL_PATH), model, optimizer)

    # ------------------------------------------------------------------------------------------------------------------

    # Store the epoch loss for training and validation sets
    train_mean_loss, val_mean_loss = [], []

    for epoch in range(NUM_EPOCHS):

        if SAVE_MODEL:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }

            # The epoch part in the filename helps save the model at each epoch
            # TODO: maybe change it to save the model at every logging_interval?
            save_checkpoint(checkpoint, output_folder=OUTPUT_FOLDER, filename=f"my_checkpoint.pth.tar-epoch={epoch}")

        # THIS WAS JUST FOR DUMMY SAKE
        # if LOAD_MODEL:
        #     step = load_checkpoint(torch.load(f'{OUTPUT_FOLDER}my_checkpoint.pth.tar-epoch={epoch}'), DUMMY_model, DUMMY_optimizer)

        # --TRAINING TIME----------------------------------------------------------------------------------------------
        # Set the model to train mode
        model.train()
        train_batch_loss = []

        print(f"Training epoch: {epoch}")
        for idx, (imgs, captions) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
            imgs = imgs.to(DEVICE)
            captions = captions.to(DEVICE)

            outputs = model(imgs, captions[:-1])  # Want the model to predict the END token so we don't send the last one in
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))

            # print(outputs)
            writer.add_scalar("Training loss", loss.item(), global_step=step)
            step += 1

            # Back prop
            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()

            # Save batch loss for training
            train_batch_loss.append(loss.item())

            # Clear GPU cache
            del idx, imgs, captions
            torch.cuda.empty_cache()

        # Compute train epoch loss
        train_mean_loss.append(np.mean(train_batch_loss))

        # --VALIDATION TIME---------------------------------------------------------------------------------------------

        # Set the model to evaluation mode
        model.eval()
        val_batch_loss = []

        print(f"Validation epoch: {epoch}")
        for idx, (imgs, captions) in tqdm(enumerate(val_loader), total=len(val_loader), leave=False):
            imgs = imgs.to(DEVICE)
            captions = captions.to(DEVICE)

            with torch.no_grad():
                outputs = model(imgs, captions[:-1])  # Want the model to predict the END token so we don't send the last one in
                loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))

            writer.add_scalar("Validation loss", loss.item(), global_step=step)

            # Save batch loss for validation
            val_batch_loss.append(loss.item())

            del idx, imgs, captions

        # Compute validation epoch loss
        val_mean_loss.append(np.mean(val_batch_loss))

        # Clear GPU cache
        torch.cuda.empty_cache()

        # Save the training and validation loss
        loss_df = pd.DataFrame({"train_mean_loss": train_mean_loss, "val_mean_loss": val_mean_loss})
        loss_csvfile = f"{OUTPUT_FOLDER}Loss.csv"
        loss_df.to_csv(loss_csvfile)


if __name__ == "__main__":
    train()
