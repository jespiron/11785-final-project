import torch
import gc
from common import device, eval, get_last_checkpoint
from network import config, model, Dataset

LAST_CHECKPOINT_PATH, RUNID, LAST_EPOCH = get_last_checkpoint()
if LAST_CHECKPOINT_PATH is None:
    print("Unable to find last check point")
    exit()
print("Checkpoint: ", LAST_CHECKPOINT_PATH)
print("Epoches: ", LAST_EPOCH)

torch.cuda.empty_cache()
gc.collect()

model.load_state_dict(torch.load(LAST_CHECKPOINT_PATH,weights_only=True))

# Create a dataset object using the AudioDataset class for the validation data
val_data = Dataset('/content/11785-f24-hw1p2/dev-clean', partition= "dev-clean")

val_loader = torch.utils.data.DataLoader(
    dataset     = val_data,
    num_workers = 2,
    batch_size  = config['batch_size'],
    pin_memory  = True,
    shuffle     = False
)
print("Validation dataset samples = {}, batches = {}".format(val_data.__len__(), len(val_loader)))

# before we test, we validate
print(eval(model, val_loader))
