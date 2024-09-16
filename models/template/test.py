import torch
import gc
from tqdm.auto import tqdm
from common import device, get_last_checkpoint
from network import config, model, TestDataset

# the best epoch is 149
LAST_CHECKPOINT_PATH, RUNID, LAST_EPOCH = get_last_checkpoint()
if LAST_CHECKPOINT_PATH is None:
    print("Unable to find last check point")
    exit()

print("Checkpoint: ", LAST_CHECKPOINT_PATH)
print("Epoches: ", LAST_EPOCH)

model.load_state_dict(torch.load(LAST_CHECKPOINT_PATH,weights_only=True))

# Create a dataset object using the AudioTestDataset class for the test data (FOR SUBMISSIONS)
test_data = TestDataset('/content/11785-f24-hw1p2/test-clean', phonemes = PHONEMES, context=config['context'], partition= "test-clean")

# Define dataloaders for train, val and test datasets
# Dataloaders will yield a batch of frames and phonemes of given batch_size at every iteration
# We shuffle train dataloader but not val & test dataloader. Why?

test_loader = torch.utils.data.DataLoader(
    dataset     = test_data,
    num_workers = 2,
    batch_size  = config['batch_size'],
    pin_memory  = True,
    shuffle     = False
)

print("Test dataset samples = {}, batches = {}".format(test_data.__len__(), len(test_loader)))

def test(model, dataloader):
    model.eval()

    ### List to store predicted phonemes of test data
    test_predictions = []

    ### Which mode do you need to avoid gradients?
    with torch.inference_mode():

        for i, input in enumerate(tqdm(dataloader)):
            input   = input.to(device)
            logits  = model(input)
    pass

predictions = test(model, test_loader)
