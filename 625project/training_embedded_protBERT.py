from Bio import SeqIO
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel, AdamW
import time
import gc

# ...
#gpu_index = 1
#torch.cuda.set_device(gpu_index)
#torch.cuda.empty_cache()
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
train_fasta_file_path ="../train_sequences.fasta"
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained("/root/autodl-fs/.sys/Tokenizer/")
model = AutoModel.from_pretrained("/root/autodl-fs/.sys/Model/").to(device)
# Wrap the model with nn.DataParallel if there are multiple GPUs
#if torch.cuda.device_count() > 1:
#    print("Using", torch.cuda.device_count(), "GPUs.")
#    model = nn.DataParallel(model)

# Move the model to the device
model = model.to(device)

MAX_LEN = 1024
#MAX_LEN = 2048
count = 0
t_initial = time.time()
list_lengths = []
list_times = []
train_sequences = SeqIO.parse(train_fasta_file_path, "fasta")
final_array = []

for seq in train_sequences:
    t_start = time.time()
    #sequence_example = seq[:MAX_LEN]
    list_lengths.append(len(seq))
    len_tmp = len(seq)
    sequence_str = ' '.join(list(seq))
    encoded_input = tokenizer(sequence_str, return_tensors='pt', max_length=MAX_LEN, truncation=True).to(device)
    output = model(**encoded_input, output_hidden_states=True)
    t_now = time.time() - t_start
    t_total = time.time() - t_initial
    list_times.append(t_start)
    last_hidden_state = output['hidden_states'][-1]
    prot_id = seq.id
    if count % 500 == 0:
        print(count, prot_id, 'len:', len_tmp, 'seconds passed:', np.round(t_now, 2), 'total', np.round(t_total, 2))

    temp_array = last_hidden_state[:, 0][0].detach().cpu().numpy()
    final_array.append(temp_array)

    count += 1
    gc.collect()

    if count % 10_000 == 0:
        output_path = './training_1024/embed_' + str(MAX_LEN) +'_count_'+ str(
            count) + '_prot.npy'
        np.save(output_path, np.array(final_array))
        final_array = []

# Save the final results
output_path = './training_1024/embed' + str(MAX_LEN) + '_count_' + str(
    count) + '_prot.npy'
np.save(output_path, np.array(final_array))
