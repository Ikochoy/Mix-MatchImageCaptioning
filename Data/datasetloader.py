# work on pytroch dataset loader -- Natalie
#inspired by Kaggle: https://www.kaggle.com/code/mdteach/torch-data-loader-flicker-8k/notebook


import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision import transforms
from .vocab import Vocabulary
import os
from PIL import Image


class Flickr8kDataset(Dataset):
    '''
    root_dir: flickr dataset folder/Images
    captions_file: flickr dataset location/captions.txt
    '''
    def __init__(self, root_dir, captions_file,
                 transform=T.Compose([]),
                 freq_threshold=5):

        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transforms.Compose([
          transforms.Resize(224),
          transforms.CenterCrop(224),
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.imgs = self.df["image"]
        self.captions = self.df["caption"]

        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocab(self.captions.to_list())


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        caption = self.captions[idx]
        img_name = self.imgs[idx]
        img_location = os.path.join(self.root_dir, img_name)
        img = Image.open(img_location).convert("RGB")

        # apply the transfromation to the image
        if self.transform is not None:
            img = self.transform(img)

        # numericalize the caption text
        caption_vec = []
        caption_vec += [self.vocab.stoi["<SOS>"]]
        caption_vec += self.vocab.numericalize(caption)
        caption_vec += [self.vocab.stoi["<EOS>"]]

        #returns image at index idx + corresponding caption
        return img, torch.tensor(caption_vec)



