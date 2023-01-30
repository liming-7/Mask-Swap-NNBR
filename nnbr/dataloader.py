from audioop import reverse
from operator import pos
import pickle
import json
from turtle import position
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils
from torch import nn
import random


class SeqDataset(Dataset):

    def __init__(self, config, type=None):

        super().__init__()
        self.dataset = config['dataset']
        self.data_path = config['data_path']
        self.foldk_path = config['foldk_path'] # use this instead of random seed for better analysis
        self.foldk = config['foldk']
        self.max_seq_length = config['max_seq_length']
        self.bask_mask_ratio = config['bask_mask_ratio']
        self.nbr_type = config['nbr_type']

        with open(self.data_path, 'r') as f:
            data = json.load(f)
        
        with open(self.foldk_path, 'r') as f:
            foldk_data = json.load(f)
        
        self.user_list = []
        self.item_seq_list = []
        self.item_position_list = []
        self.item_seq_length_list = []
        self.pos_item_list = []
        self.history_item_list = []

        if  type == 'train-nbr':
            users = foldk_data['train']
            for usr_id in users:
                bask_seq = data[usr_id] # 
                h_bask_seq = bask_seq[:-1]
                history_item = list(set([item for bask in h_bask_seq for item in bask]))
                h_bask_seq = self.truncate(h_bask_seq)
                pos_item = bask_seq[-1] # pos_item
                pos_expl_item = list(set(pos_item)-set(history_item))
                if self.nbr_type == 'explore':
                    h_bask_seq.append(pos_expl_item)
                elif self.nbr_type == 'all':
                    h_bask_seq.append(pos_item)
                if (self.nbr_type=='explore') and (len(pos_expl_item) == 0):
                    continue
                else:
                    item_seq = []
                    position_seq = []
                    for bask_ind, bask in enumerate(h_bask_seq):
                        item_seq.extend(bask)
                        position_seq.extend([bask_ind+1 for _ in range(len(bask))]) # position id start from 1 pad with 0

                    self.user_list.append(int(usr_id))
                    self.item_seq_list.append(torch.tensor(item_seq))
                    self.item_position_list.append(torch.tensor(position_seq))
                    self.item_seq_length_list.append(len(item_seq))
                    self.pos_item_list.append(torch.tensor(pos_item))
                    self.history_item_list.append(torch.tensor(history_item))

        else:
            users = foldk_data[type]
            for usr_id in users:
                bask_seq = data[usr_id] # 
                h_bask_seq = bask_seq[:-1]
                history_item = list(set([item for bask in h_bask_seq for item in bask]))
                h_bask_seq = self.truncate(h_bask_seq)
                pos_item = bask_seq[-1] # pos_item
                pos_expl_item = list(set(pos_item)-set(history_item))
                if (self.nbr_type=='explore') and (len(pos_expl_item) == 0):
                    continue
                else:
                    item_seq = []
                    position_seq = []
                    for bask_ind, bask in enumerate(h_bask_seq):
                        item_seq.extend(bask)
                        position_seq.extend([bask_ind+1 for _ in range(len(bask))]) # position id start from 1 pad with 0

                    self.user_list.append(int(usr_id))
                    self.item_seq_list.append(torch.tensor(item_seq))
                    self.item_position_list.append(torch.tensor(position_seq))
                    self.item_seq_length_list.append(len(item_seq))
                    self.pos_item_list.append(torch.tensor(pos_item))
                    self.history_item_list.append(torch.tensor(history_item))


    def __getitem__(self, index):
        user = self.user_list[index]
        item_seq = self.item_seq_list[index]
        item_position_seq = self.item_position_list[index]
        item_seq_length = self.item_seq_length_list[index]
        pos_item = self.pos_item_list[index]
        history_item = self.history_item_list[index] # placeholder, do not use bpr now
        return user, item_seq, item_position_seq, item_seq_length, pos_item, history_item
    
    def __len__(self):
        return len(self.item_seq_list)

    def truncate(self, bask_seq):
        item_cnt = 0
        bask_truncate_ind = 0
        while item_cnt <= self.max_seq_length:
            bask_truncate_ind += 1
            if bask_truncate_ind>len(bask_seq):
                break
            else:
                item_cnt += len(bask_seq[-bask_truncate_ind])
        if bask_truncate_ind<len(bask_seq):
            bask_seq = bask_seq[-bask_truncate_ind:].copy()
        return bask_seq

# dynamic pad
def collate_fn(batch_data):
    ret = list()
    for idx, data_list in enumerate(zip(*batch_data)):
        if isinstance(data_list[0], torch.Tensor) and idx==1:
            # data_list.sort(key=lambda data: len(data), reverse=True)
            item_seq = rnn_utils.pad_sequence(data_list, batch_first=True, padding_value=0)
            # print(item_seq)
            ret.append(item_seq)
        elif isinstance(data_list[0], torch.Tensor) and idx==2:
            position_seq = rnn_utils.pad_sequence(data_list, batch_first=True, padding_value=0)
            ret.append(position_seq)
        elif isinstance(data_list[0], torch.Tensor) and idx==4:
            pos_item_seq = rnn_utils.pad_sequence(data_list, batch_first=True, padding_value=-1)
            ret.append(pos_item_seq) 
        elif isinstance(data_list[0], torch.Tensor) and idx==5:
            history_item_seq = rnn_utils.pad_sequence(data_list, batch_first=True, padding_value=-1)
            ret.append(history_item_seq)
        else:
            # print(data_list)
            ret.append(torch.tensor(data_list))
    # (user, item_seq, item_seq_length, pos_item, neg_item) --> tensor format
    return tuple(ret) 

# pad to the max lenth 50 or 100, 
def collate_fn_max_len50(batch_data):
    ret = list()
    for idx, data_list in enumerate(zip(*batch_data)):
        if isinstance(data_list[0], torch.Tensor) and idx==1:
            # print(data_list[0])
            # print(nn.ConstantPad1d((0, 50-data_list[0].shape[0]), 0)(data_list[0]))
            data_list = list(data_list)
            data_list[0] = nn.ConstantPad1d((0, 50-data_list[0].shape[0]), 0)(data_list[0])
            data_list = tuple(data_list)
            # data_list.sort(key=lambda data: len(data), reverse=True)
            item_seq = rnn_utils.pad_sequence(data_list, batch_first=True, padding_value=0)
            # print(item_seq)
            ret.append(item_seq)
        else:
            ret.append(torch.tensor(data_list))
    # (user, item_seq, item_seq_length, pos_item, neg_item) --> tensor format
    return tuple(ret) 

def get_dataloader(config, d_type=None):
    dataset = SeqDataset(config, type=d_type)
    print(f'{d_type} data length -> {len(dataset)}')
    if config['method'] == 'repeatnet':
        data_loader = DataLoader(dataset=dataset, batch_size=config['batch_size'], shuffle=True, drop_last=False, collate_fn=collate_fn_max_len50)
    else:
        data_loader = DataLoader(dataset=dataset, batch_size=config['batch_size'], shuffle=True, drop_last=False, collate_fn=collate_fn)
    return data_loader