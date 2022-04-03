import torch
from torch.nn.utils.rnn import pad_sequence


class CollatePadding:
    '''
    Pass this class into the DataLoader as the collate_fn
    in order to pad the images and captions properly

    batch_First = True for when passing into Dataloader, set to False otherwise
    '''

    def __init__(self, pad_idx, batch_first=True):

        self.pad_idx= pad_idx
        self.batch_first = batch_first

    def __call__(self, batch):

        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)

        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=self.batch_first,
                               padding_value = self.pad_idx)

        return imgs, targets
