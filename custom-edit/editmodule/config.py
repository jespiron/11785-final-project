import torch 
from diffusers import DiffusionPipeline, SchedulerMixin
from dataclasses import dataclass
from transformers import CLIPTokenizer

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# Constants
MY_TOKEN = ''
LOW_RESOURCE = False 
NUM_DDIM_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77

# Changable parameters
@dataclass
class Config:
    ldm_stable: DiffusionPipeline
    tokenizer: CLIPTokenizer
    scheduler: SchedulerMixin

config = Config(None, None, None)