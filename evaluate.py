import argparse
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from models import Autoformer, DLinear, TimeLLM
from data_provider.data_factory import data_provider
import time
import random
import numpy as np
import os
os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali, load_content
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from data_provider.data_factory import data_provider
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt



fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)



class Args:
    def __init__(self):
        self.task_name = 'long_term_forecast'
        self.is_training = 1
        self.model_id = 'T1_DM_12_12'
        self.model_comment = 'TimeLLM-T1DM'
        self.model = 'TimeLLM'
        self.seed = 2021
        self.data = 'T1DM'
        self.root_path = './dataset/T1DM_Fake/'
        self.train_data_path = '570-ws-training.csv'
        self.test_data_path = '570-ws-training.csv'
        self.features = 'S'
        self.target = '_value'
        self.loader = 'modal'
        self.freq = 'h'
        self.checkpoints = './checkpoints/'
        self.seq_len = 12 #6
        self.label_len = 6 #  6
        self.pred_len = 12 #6  9 12
        self.enc_in = 9
        self.dec_in = 9
        self.c_out = 9
        self.d_model = 32
        self.n_heads = 8  # Typically set by your model configuration
        self.e_layers = 2  # Typically set by your model configuration
        self.d_layers = 1  # Typically set by your model configuration
        self.d_ff = 32
        self.moving_avg = 25  # Assume default if not specified in the script
        self.factor = 1
        self.dropout = 0.1  # Assume default if not specified
        self.embed = 'timeF'  # Assume default if not specified
        self.activation = 'gelu'  # Assume default if not specified
        self.output_attention = False  # Assume default if not specified
        self.patch_len = 16  # Assume default if not specified
        self.stride = 8  # Assume default if not specified
        self.prompt_domain = 0  # Assume default if not specified
        self.llm_model = 'GPT2'
        self.llm_dim = 768
        self.num_workers = 10  # Default setting
        self.itr = 1
        self.train_epochs = 2
        self.align_epochs = 10  # Assume default if not specified
        self.batch_size = 7
        self.eval_batch_size = 8  # Assume default if not specified
        self.patience = 10  # Assume default if not specified
        self.learning_rate = 0.001
        self.des = 'Exp'
        self.loss = 'MSE'  # Assume default if not specified
        self.lradj = 'type1'  # Assume default if not specified
        self.pct_start = 0.2  # Assume default if not specified
        self.use_amp = False  # Assume default based on your environment capabilities
        self.llm_layers = 32
        self.percent = 100  # Assume default if not specified

# Instantiate the Args
args = Args()

path = 'checkpoints/long_term_forecast_T1_DM_12_12_TimeLLM_T1DM_features-S_seq-12_lr-0.001_TimeLLM-T1DM_20250107-191341.pt'
model = TimeLLM.Model(args).float()
model.load_state_dict(torch.load(path), strict=False)
model.eval()
print(model)

train_data, train_loader = data_provider(args, 'train')
vali_data, vali_loader = data_provider(args, 'val')
test_data, test_loader = data_provider(args, 'test')

time_now = time.time()

train_steps = len(train_loader)
early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)

trained_parameters = []

for p in model.parameters():
    if p.requires_grad is True:
        trained_parameters.append(p)

model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

if args.lradj == 'COS':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)

else:
    scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                        steps_per_epoch=train_steps,
                                        pct_start=args.pct_start,
                                        epochs=args.train_epochs,
                                        max_lr=args.learning_rate)


train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
                train_loader, vali_loader, test_loader, model, model_optim, scheduler)
print("dataloaders are loaded.")

# Setup scaler for managing precision
scaler = GradScaler()
predictions = list()
true_labels = list()
with torch.no_grad():  # No need to compute gradients during inference
    for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
        batch_x = batch_x.float().to(accelerator.device)
        batch_y = batch_y.float().to(accelerator.device)
        batch_x_mark = batch_x_mark.float().to(accelerator.device)
        batch_y_mark = batch_y_mark.float().to(accelerator.device)
        # Prepare decoder input as zeros initially, similar to training phase setup
        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).to(accelerator.device)
        dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1)

        # Using autocast for automatic mixed precision
        with autocast():
            if args.output_attention:
                output, _ = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                output = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            output = output[:, -args.pred_len:, :]  # Adjust based on the output dimensions if needed

        predictions.append(output.cpu().numpy())
        true_labels.append(batch_y[:, -args.pred_len:, :].cpu().numpy())
# Convert list of arrays to a single numpy array
predictions = np.concatenate(predictions, axis=0)
true_labels = np.concatenate(true_labels, axis=0)
print("Predictions are generated.")

plt.figure(figsize=(12, 6))
plt.plot(true_labels[:, 0], label='Actual Data', color='blue')  # Adjust indexing based on your data shape
plt.plot(predictions[:, 0], label='Predicted Data', color='red')  # Adjust indexing based on your data shape
plt.title('Time Series Forecasting')
plt.xlabel('Time Steps')
plt.ylabel('Values')
plt.legend()

# Dynamically construct the filename for the plot
plot_filename = f"plots/{args.task_name}_{args.model_id}_{args.model}_{args.data}_features-{args.features}_seq-{args.seq_len}_lr-{args.learning_rate}_{args.model_comment}.png"

# Ensure the directory exists
os.makedirs(os.path.dirname(plot_filename), exist_ok=True)

# Save the plot
plt.savefig(plot_filename)
print(f"Plot saved at {plot_filename}")

plt.show()