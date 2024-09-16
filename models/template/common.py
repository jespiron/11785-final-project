import torch
import numpy as np
import sklearn
import zipfile
import pandas as pd
from tqdm.auto import tqdm
import os
from glob import glob
import datetime
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", device)


def train(model, dataloader, optimizer, criterion):

    model.train()
    tloss, tacc = 0, 0 # Monitoring loss and accuracy
    batch_bar   = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc='Train')

    for i, (frames, phonemes) in enumerate(dataloader):

        ### Initialize Gradients
        optimizer.zero_grad()

        ### Move Data to Device (Ideally GPU)
        frames      = frames.to(device)
        phonemes    = phonemes.to(device)

        ### Forward Propagation
        logits  = model(frames)

        ### Loss Calculation
        loss    = criterion(logits, phonemes)

        ### Backward Propagation
        loss.backward()

        ### Gradient Descent
        optimizer.step()

        tloss   += loss.item()
        tacc    += torch.sum(torch.argmax(logits, dim= 1) == phonemes).item()/logits.shape[0]

        batch_bar.set_postfix(loss="{:.04f}".format(float(tloss / (i + 1))),
                              acc="{:.04f}%".format(float(tacc*100 / (i + 1))))
        batch_bar.update()

        ### Release memory
        del frames, phonemes, logits
        torch.cuda.empty_cache()

    batch_bar.close()
    tloss   /= len(dataloader)
    tacc    /= len(dataloader)

    return tloss, tacc

criterion = torch.nn.CrossEntropyLoss() # Defining Loss function.
def eval(model, dataloader):

    model.eval() # set model in evaluation mode
    vloss, vacc = 0, 0 # Monitoring loss and accuracy
    batch_bar   = tqdm(total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc='Val')

    for i, (frames, phonemes) in enumerate(dataloader):

        ### Move data to device (ideally GPU)
        frames      = frames.to(device)
        phonemes    = phonemes.to(device)

        # makes sure that there are no gradients computed as we are not training the model now
        with torch.inference_mode():
            ### Forward Propagation
            logits  = model(frames)
            ### Loss Calculation
            loss    = criterion(logits, phonemes)

        vloss   += loss.item()
        vacc    += torch.sum(torch.argmax(logits, dim= 1) == phonemes).item()/logits.shape[0]

        # Do you think we need loss.backward() and optimizer.step() here?

        batch_bar.set_postfix(loss="{:.04f}".format(float(vloss / (i + 1))),
                              acc="{:.04f}%".format(float(vacc*100 / (i + 1))))
        batch_bar.update()

        ### Release memory
        del frames, phonemes, logits
        torch.cuda.empty_cache()

    batch_bar.close()
    vloss   /= len(dataloader)
    vacc    /= len(dataloader)

    return vloss, vacc

def get_checkpoint_name(epoch: int) -> str:
    return "epoch-" + str(epoch).zfill(5)

def get_experiment_dir():
    return os.path.dirname(os.path.abspath(__file__))

def get_last_checkpoint(requestedEpoch=None):
    result = None
    runid = None
    epoch = 0

    wandb_dir=get_experiment_dir() + "/wandb"
    if os.path.exists(wandb_dir):
        for run in sorted(glob(wandb_dir + "/*run-*")):
            epoches = sorted(glob(run + "/files/epoch-*"))
            if requestedEpoch is not None:
                # loop through epoches to find the requested epoch
                for e in epoches:
                    tmp = int(os.path.basename(e).split('-')[1]) + 1
                    if tmp == requestedEpoch:
                        result = e
                        runid = os.path.basename(run).split('-')[-1]
                        epoch = tmp
                        break
                if result is not None:
                    break
            elif len(epoches) > 0:
                tmp = int(os.path.basename(epoches[-1]).split('-')[1]) + 1
                if tmp > epoch:
                    result = epoches[-1]
                    runid = os.path.basename(run).split('-')[-1]
                    epoch = tmp
    return (result, runid, epoch)

