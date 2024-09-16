import torch

# use per-utterance padding
from data_set import Dataset, TestDataset

config = {
    'batch_size'    : 1024,
    'context'       : 30,
    'init_lr'       : 4e-3,
    'weight_decay'  : 0.01,
    'optimizer'     : 'AdamW',
    'architecture'  : '1-2-3'

    # Add more as you need them - e.g dropout values, weight decay, scheduler parameters
}

class Network(torch.nn.Module):

    def __init__(self, input_size, output_size):

        super(Network, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1024, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, output_size)
        )

    def forward(self, x):
        out = self.model(x)
        return out

device = 'cuda' if torch.cuda.is_available() else 'cpu'
INPUT_SIZE  = 0xdeadbeef
OUTPUT_SIZE = 0xdeadbeef
model       = Network(INPUT_SIZE, OUTPUT_SIZE).to(device)
optimizer = torch.optim.AdamW(
            model.parameters(),
            lr= config['init_lr'],
            weight_decay=config['weight_decay'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

print("Batch size     : ", config['batch_size'])
print("Context        : ", config['context'])
print("Input size     : ", INPUT_SIZE)
