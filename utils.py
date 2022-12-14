import torch
import torchvision.transforms as transforms
from PIL import Image


def save_checkpoint(state, output_folder, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, f"{output_folder}{filename}")


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step
