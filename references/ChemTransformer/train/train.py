import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.transformer import TransformerModel
from utils import ReactionSequenceDataset, train, validation
import sys
import yaml
import time
import shutil

config_path = sys.argv[1]
config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
model_conf = config['model']
data_conf = config['data']
train_conf = config['training']
output_conf = config['output']

# dataset & dictionary
data_root = data_conf['data_root']
dictionary_path = os.path.join(data_root, 'dictionary.pkl')
train_path = os.path.join(data_root, 'train.npz')
val_path = os.path.join(data_root, 'val.npz')
log_path = os.path.join(data_root, 'log.txt')
token2index,_ = pickle.load(open(dictionary_path, 'rb'))
pad_idx = token2index['<PAD>']
vocab_size = len(token2index)
batch_size = data_conf['batch_size']

# model & training
d_model = model_conf['d_model']
nhead = model_conf['n_head']
num_layers = model_conf['n_layer']
dim_feedforward = model_conf['d_ff']
dropout = model_conf['dropout']
epochs = train_conf['max_epoch']
ini_lr, term_lr = train_conf['max_lr'], train_conf['min_lr']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(output_conf['ckpt_dir'], exist_ok=True)
shutil.copy(config_path, os.path.join(output_conf['ckpt_dir'], 'config.yaml'))

# dataset
train_dataset = ReactionSequenceDataset(train_path)
val_dataset = ReactionSequenceDataset(val_path)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)

model = TransformerModel(vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=token2index['<PAD>'])
optimizer = optim.Adam(model.parameters(), lr=ini_lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500000, eta_min=term_lr)

with open(log_path, 'w') as f:
    f.write('Ep, Train Loss, Val Loss, Train Token Acc, Train Seq Acc, Val Token Acc, Val Seq Acc\n')

# main loop
for epoch in range(epochs):
    print(f'Training epoch {epoch+1}/{epochs}')
    start_time = time.time()

    avg_loss, train_token_accuracy, train_sequence_accuracy = \
                train(model, train_loader, optimizer, criterion, vocab_size, device)
    
    val_loss, val_token_accuracy, val_sequence_accuracy = \
                validation(model, test_loader, criterion, vocab_size, device)

    with open(log_path, 'a') as f:
        f.write(f'{epoch+1},{avg_loss:.4f},{val_loss:.4f},{train_token_accuracy:.4f}, {train_sequence_accuracy:.4f},{val_token_accuracy:.4f},{val_sequence_accuracy:.4f}\n')

    if (epoch+1) % output_conf['ckpt_interval'] == 0:
        torch.save(model.state_dict(), os.path.join(output_conf['ckpt_dir'], f'ep{(epoch+1):03f}.pt'))

    print('Time Elapsed:', time.time() - start_time)