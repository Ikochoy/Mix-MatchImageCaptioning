import pandas as pd
import numpy as np

train_split = 'Flickr_8k.trainImages.txt'
val_split = 'Flickr_8k.devImages.txt'
test_split = 'Flickr_8k.testImages.txt'
captions = 'flickr8k/captions.txt'

train_df = pd.read_csv(train_split)
val_df = pd.read_csv(val_split)
test_df = pd.read_csv(test_split)
captions_df = pd.read_csv(captions)

train_captions = {'image': [], 'caption': []}
val_captions = {'image': [], 'caption': []}
test_captions = {'image': [], 'caption': []}

train_imgs, val_imgs, test_imgs = [], [], []
missing = []

# print(captions_df)
for index, row in captions_df.iterrows():
    img_id = row["image"]
    caption = row["caption"]
    print('The value of A: {}, B: {}'.format(img_id, caption))

    if img_id in train_df.image.values:
        # In training set
        train_captions['image'].append(img_id)
        train_captions['caption'].append(caption)

        train_imgs.append(img_id)

    elif img_id in val_df.image.values:
        # In val set
        val_captions['image'].append(img_id)
        val_captions['caption'].append(caption)

        val_imgs.append(img_id)

    elif img_id in test_df.image.values:
        # In test set
        test_captions['image'].append(img_id)
        test_captions['caption'].append(caption)

        test_imgs.append(img_id)

    else:
        print("OH no what happened")
        missing.append(img_id)

train_captions_df = pd.DataFrame(train_captions)  # move this into training_loop function
train_captions_csvfile = "train_captions.txt"

val_captions_df = pd.DataFrame(val_captions)  # move this into training_loop function
val_captions_csvfile = "val_captions.txt"

test_captions_df = pd.DataFrame(test_captions)  # move this into training_loop function
test_captions_csvfile = "test_captions.txt"

train_captions_df.to_csv(train_captions_csvfile, index=False)
val_captions_df.to_csv(val_captions_csvfile, index=False)
test_captions_df.to_csv(test_captions_csvfile, index=False)

print(len(missing))
print(len(set(train_imgs)), len(set(val_imgs)), len(set(test_imgs)))

