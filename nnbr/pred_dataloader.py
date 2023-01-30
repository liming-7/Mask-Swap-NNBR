from audioop import reverse
from operator import pos
import pickle
import json
from turtle import position
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils
from torch import nn


class SeqDataset(Dataset):

    def __init__(self, config, type=None):

        super().__init__()
        self.data_path = config['data_path']
        self.foldk_path = config['foldk_path'] # use this instead of random seed for better analysis
        self.foldk = config['foldk']
        self.max_seq_length = config['max_seq_length']
        # self.filter_repeat = config['filter_repeat'] # for train, this is to manuplate the dataset its own, not to attack the dataset.
        
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        
        with open(self.foldk_path, 'r') as f:
            foldk_data = json.load(f)
        
        users = foldk_data[type]
        self.user_list = []
        self.item_seq_list = []
        self.item_position_list = []
        self.item_seq_length_list = []
        self.pos_item_list = []

        for usr_id in users:
            if type == 'train':
                bask_seq = data[usr_id] # this is for bert, other should notice this.
            else:
                bask_seq = data[usr_id][:-1]
            # bask_seq = data[usr_id] # for bert we mask the last item when we perform predicting
            item_cnt = 0
            bask_truncate_ind = 0
            while item_cnt <= self.max_seq_length:
                bask_truncate_ind += 1
                if bask_truncate_ind>len(bask_seq):
                    break
                else:
                    item_cnt += len(bask_seq[-bask_truncate_ind])
            if bask_truncate_ind<len(bask_seq):
                bask_seq = bask_seq[-bask_truncate_ind:]
            item_seq = []
            position_seq = []
            pos_item = data[usr_id][-1]
            bask_ind = 1
            for bask in bask_seq:
                item_seq = item_seq + bask
                position_seq = position_seq + [bask_ind for _ in range(len(bask))]
                bask_ind += 1

            self.user_list.append(int(usr_id))
            self.item_seq_list.append(torch.tensor(item_seq))
            self.item_position_list.append(torch.tensor(position_seq))
            self.item_seq_length_list.append(len(item_seq))
            self.pos_item_list.append(torch.tensor(pos_item))


    def __getitem__(self, index):
        user = self.user_list[index]
        item_seq = self.item_seq_list[index]
        item_position_seq = self.item_position_list[index]
        item_seq_length = self.item_seq_length_list[index]
        pos_item = self.pos_item_list[index]
        neg_item = 0 # placeholder, do not use bpr now
        return user, item_seq, item_position_seq, item_seq_length, pos_item, neg_item
    
    def __len__(self):
        return len(self.item_seq_list)

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