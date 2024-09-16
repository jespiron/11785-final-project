import torch
import wandb
import gc
import os
from common import device, eval, train, get_checkpoint_name, get_last_checkpoint
from torchsummaryX import summary

LAST_CHECKPOINT_PATH, RUNID, LAST_EPOCH = get_last_checkpoint()
from network import config, model, INPUT_SIZE, optimizer, scheduler, Dataset

summary(model, torch.zeros([config['batch_size'], INPUT_SIZE]).to(device))

# Create a dataset object using the AudioDataset class for the training data
train_data = Dataset('/content/11785-f24-hw1p2/train-clean-100', partition= "train-clean-100")

# Create a dataset object using the AudioDataset class for the validation data
val_data = Dataset('/content/11785-f24-hw1p2/dev-clean', partition= "dev-clean")

train_loader = torch.utils.data.DataLoader(
    dataset     = train_data,
    num_workers = 4,
    batch_size  = config['batch_size'],
    pin_memory  = True,
    shuffle     = True
)

val_loader = torch.utils.data.DataLoader(
    dataset     = val_data,
    num_workers = 2,
    batch_size  = config['batch_size'],
    pin_memory  = True,
    shuffle     = False
)

print("Train dataset samples = {}, batches = {}".format(train_data.__len__(), len(train_loader)))
print("Validation dataset samples = {}, batches = {}".format(val_data.__len__(), len(val_loader)))
# Testing code to check if your data loaders are working
for i, data in enumerate(train_loader):
    frames, phoneme = data
    print(frames.shape, phoneme.shape)
    break

wandb.login(key="2cae5ca7549879f7fc772a096ff77b469f6bd3dc") #API Key is in your wandb account, under settings (wandb.ai/settings)

# Create your wandb run
wandb_name=os.uname()[1] + ":" + os.path.basename(os.path.dirname(os.path.abspath(__file__)))
run = wandb.init(
    name    = wandb_name, ### Wandb creates random run names if you skip this field, we recommend you give useful names
    dir     = os.path.dirname(os.path.abspath(__file__)),
    reinit  = RUNID is None,
    id      = RUNID,
    resume  = None if RUNID is None else "must",
    project = "hw1p2", ### Project should be created in your wandb account
    config  = config ### Wandb Config for your run
)

### Save your model architecture as a string with str(model)
model_arch  = str(model)

### Save it in a txt file
arch_file   = open("model_arch.txt", "w")
file_write  = arch_file.write(model_arch)
arch_file.close()

### log it in your wandb run with wandb.save()
wandb.save('model_arch.txt')

# Iterate over number of epochs to train and evaluate your model
# This generates a very large DB
# wandb.watch(model, log="all")

# Restore checkpoint if exists
if LAST_CHECKPOINT_PATH is not None:
    model.load_state_dict(torch.load(LAST_CHECKPOINT_PATH,weights_only=True))

criterion = torch.nn.CrossEntropyLoss() # Defining Loss function.
# We use CE because the task is multi-class classification

torch.cuda.empty_cache()
gc.collect()


epochs = 500
epoch_start = LAST_EPOCH
for epoch in range(epoch_start, epochs):
    print("\nEpoch {}/{}".format(epoch+1, epochs))

    curr_lr                 = float(optimizer.param_groups[0]['lr'])
    train_loss, train_acc   = train(model, train_loader, optimizer, criterion)
    val_loss, val_acc       = eval(model, val_loader)
    if scheduler is not None:
        scheduler.step(val_loss)

    print("\tTrain Acc {:.04f}%\tTrain Loss {:.04f}\t Learning Rate {:.07f}".format(train_acc*100, train_loss, curr_lr))
    print("\tVal Acc {:.04f}%\tVal Loss {:.04f}".format(val_acc*100, val_loss))

    ### Log metrics at each epoch in your run
    # Optionally, you can log at each batch inside train/eval functions
    # (explore wandb documentation/wandb recitation)
    wandb.log({'train_acc': train_acc*100, 'train_loss': train_loss,
               'val_acc': val_acc*100, 'valid_loss': val_loss, 'lr': curr_lr})

    ### Highly Recommended: Save checkpoint in drive and/or wandb if accuracy is better than your current best
    CHECKPOINT_NAME = get_checkpoint_name(epoch)
    CHECKPOINT_PATH = os.path.join(wandb.run.dir, CHECKPOINT_NAME)
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    wandb.save(CHECKPOINT_NAME)

