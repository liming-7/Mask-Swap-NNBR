import random
import math
from turtle import position
import torch
from torch import nn
import numpy as np
from recbole.model.layers import TransformerEncoder


class BERT4Expl(nn.Module):

    def __init__(self, config):
        super(BERT4Expl, self).__init__()

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['dropout_prob']
        self.attn_dropout_prob = config['dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        self.mask_type = config['mask_type']
        self.mask_ratio = config['mask_ratio']
        self.swap_ratio = config['swap_ratio']
        self.nbr_type = config['nbr_type']
        self.swap_hop = config['swap_hop']

        self.loss_type = config['loss_type']
        # self.filter_cat = config['']
        self.initializer_range = config['initializer_range']

        # dataset info
        self.n_items = config['n_items']
        self.max_seq_length = config['max_seq_length']

        # load dataset info
        self.mask_token = self.n_items
        self.mask_item_length = int(self.mask_ratio * self.max_seq_length) 

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items + 1, self.hidden_size, padding_idx=0)  # mask token add 1
        self.position_embedding = nn.Embedding(60, self.hidden_size, padding_idx=0)  # add mask_token at the last
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        # we only need compute the loss at the masked position
        try:
            assert self.loss_type in ['BPR', 'BCE']
        except AssertionError:
            raise AssertionError("Make sure 'loss_type' in ['BPR', 'BCE']!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _neg_sample(self, item_set): # choose one item not in the item_set
        item = random.randint(1, self.n_items - 1)
        while item in item_set:
            item = random.randint(1, self.n_items - 1)
        return item

    def _padding_sequence(self, sequence, max_length):
        pad_len = max_length - len(sequence)
        sequence = [0] * pad_len + sequence
        sequence = sequence[-max_length:]  # truncate according to the max_length
        return sequence

    def reconstruct_bert_train_data(self, item_seq, position_seq):
        """
        Mask item sequence for training.
        """
        device = item_seq.device
        batch_size = item_seq.size(0)

        sequence_instances = item_seq.cpu().numpy().tolist()
        # Masked Item Prediction
        # [B * Len]
        masked_item_sequence = []
        pos_items = []
        neg_items = []
        masked_index = []

        swaped_position_sequence = []
        batch_position_seq = position_seq.cpu().numpy().tolist()

        # new masking method
        for instance, position_seq in zip(sequence_instances, batch_position_seq):
            # WE MUST USE 'copy()' HERE!
            masked_sequence = instance.copy()
            pos_item = []
            neg_item = []
            index_ids = []
            
            if self.mask_type == 'select':
                item_set = set(masked_sequence)
                if len(item_set) < 2: # can not mask all items
                    continue
                mask_cnt = math.ceil(len(item_set)*self.mask_ratio)
                if mask_cnt == len(item_set):
                    mask_cnt = len(item_set)-1
                masked_item = random.sample(item_set, mask_cnt)
                for index_id, item in enumerate(instance):
                    # padding is 0, the sequence is end
                    if item == 0 and (index_id!=0):
                        break
                    if item in masked_item:
                        pos_item.append(item)
                        neg_item.append(self._neg_sample(instance))
                        masked_sequence[index_id] = self.mask_token
                        index_ids.append(index_id)

                masked_item_sequence.append(masked_sequence)
                pos_items.append(self._padding_sequence(pos_item, self.max_seq_length))
                neg_items.append(self._padding_sequence(neg_item, self.max_seq_length))
                masked_index.append(self._padding_sequence(index_ids, self.max_seq_length))

            #### Mask according to prob.
            if self.mask_type == 'random':
                for index_id, item in enumerate(instance):
                    mask_prob = random.random()
                    if mask_prob < self.mask_ratio:
                        pos_item.append(item)
                        neg_item.append(self._neg_sample(instance))
                        masked_sequence[index_id] = self.mask_token
                        index_ids.append(index_id)

                ######## original mask ratio
                masked_item_sequence.append(masked_sequence)
                pos_items.append(self._padding_sequence(pos_item, self.mask_item_length))
                neg_items.append(self._padding_sequence(neg_item, self.mask_item_length))
                masked_index.append(self._padding_sequence(index_ids, self.mask_item_length))
            
                        
            swaped_position = position_seq.copy()
            max_pos = np.max(position_seq)
            # print(max_pos)
            for index_id, position in enumerate(position_seq):
                if position == 0:
                    break
                prob = random.random()
                left_position = np.max([position-self.swap_hop, 1])
                right_position = np.min([position+self.swap_hop, max_pos])
                if prob < self.swap_ratio:
                    if position <= self.swap_hop:
                        swaped_position[index_id] =  right_position # move to right
                    else:
                        left_prob = random.random()
                        if left_prob < 0.5:
                            swaped_position[index_id] = left_position # move to left basket.
                        else:
                            swaped_position[index_id] = right_position # move to right basket.
            swaped_position_sequence.append(swaped_position)

        # [B Len]
        batch_size = len(masked_item_sequence)
        # print(len(masked_item_sequence))
        masked_item_sequence = torch.tensor(masked_item_sequence, dtype=torch.long, device=device).view(batch_size, -1)
        # [B mask_len]
        pos_items = torch.tensor(pos_items, dtype=torch.long, device=device).view(batch_size, -1)
        # [B mask_len]
        neg_items = torch.tensor(neg_items, dtype=torch.long, device=device).view(batch_size, -1)
        # [B mask_len]
        masked_index = torch.tensor(masked_index, dtype=torch.long, device=device).view(batch_size, -1)

        swaped_position_sequence = torch.tensor(swaped_position_sequence, dtype=torch.long, device=device).view(batch_size, -1)

        return masked_item_sequence, swaped_position_sequence, pos_items, neg_items, masked_index

    def reconstruct_test_data(self, item_seq, position_seq, item_seq_len): #maybe a bug here?
        """
        Add mask token at the last position according to the lengths of item_seq
        """
        padding = torch.zeros(item_seq.size(0), dtype=torch.long, device=item_seq.device)  # [B]
        item_seq = torch.cat((item_seq, padding.unsqueeze(-1)), dim=-1)  # [B max_len+1]
        position_seq = torch.cat((position_seq, padding.unsqueeze(-1)), dim=-1)
        for batch_id, last_position in enumerate(item_seq_len):
            item_seq[batch_id][last_position] = self.mask_token
            position_seq[batch_id][last_position] = position_seq[batch_id][last_position-1]+1
        return item_seq, position_seq

    def reconstruct_nbr_train_data(self, item_seq, position_seq):
        """
        Mask item sequence for training.
        """
        device = item_seq.device
        batch_size = item_seq.size(0)

        sequence_instances = item_seq.cpu().numpy().tolist()
        # Masked Item Prediction
        # [B * Len]
        masked_item_sequence = []
        pos_items = []
        neg_items = []
        masked_index = []

        swaped_position_sequence = []
        batch_position_seq = position_seq.cpu().numpy().tolist()

        # new masking method
        for instance, position_seq in zip(sequence_instances, batch_position_seq):
            # WE MUST USE 'copy()' HERE!
            masked_sequence = instance.copy()
            swaped_position = position_seq.copy()

            pos_item = []
            neg_item = []
            index_ids = []
            max_pos = np.max(swaped_position)
            for index_id, (item, position) in enumerate(zip(instance, position_seq)):
                # padding is 0, the sequence is end
                if position == max_pos:
                    pos_item.append(item)
                    neg_item.append(self._neg_sample(instance))
                    masked_sequence[index_id] = self.mask_token
                    index_ids.append(index_id)

            masked_item_sequence.append(masked_sequence)
            pos_items.append(self._padding_sequence(pos_item, 50))
            neg_items.append(self._padding_sequence(neg_item, 50))
            masked_index.append(self._padding_sequence(index_ids, 50))

            swaped_position_sequence.append(swaped_position)

        # [B Len]
        batch_size = len(masked_item_sequence)
        # print(len(masked_item_sequence))
        masked_item_sequence = torch.tensor(masked_item_sequence, dtype=torch.long, device=device).view(batch_size, -1)
        # [B mask_len]
        pos_items = torch.tensor(pos_items, dtype=torch.long, device=device).view(batch_size, -1)
        # [B mask_len]
        neg_items = torch.tensor(neg_items, dtype=torch.long, device=device).view(batch_size, -1)
        # [B mask_len]
        masked_index = torch.tensor(masked_index, dtype=torch.long, device=device).view(batch_size, -1)

        swaped_position_sequence = torch.tensor(swaped_position_sequence, dtype=torch.long, device=device).view(batch_size, -1)

        return masked_item_sequence, swaped_position_sequence, pos_items, neg_items, masked_index

    
    def construct_pos_neg_sample(self, pos_items, history_items, neg_num=5):
        device = pos_items.device
        pos_items = pos_items.cpu().numpy().tolist()
        history_items = history_items.cpu().numpy().tolist()
        pos_items_list = []
        neg_items_list = []
        for p_item, h_item in zip(pos_items, history_items):
            # pos items

            p_item = set(p_item)
            p_item.discard(-1)
            if self.nbr_type == 'all':
                p_item_set = list(set(p_item))
            else:
                p_item_set = list(set(p_item)-set(h_item))
            if len(p_item_set) < neg_num:
                p_item_set_k = random.choices(p_item_set, k=neg_num)
            else:
                p_item_set_k = random.sample(p_item_set, neg_num)
            pos_items_list.append(torch.tensor(p_item_set_k))
            # neg items
            s_item_set = set(p_item)&set(h_item)
            n_item_set_k = [self._neg_sample(s_item_set) for _ in range(neg_num)]
            neg_items_list.append(torch.tensor(n_item_set_k))
        pos_items_tensor = torch.stack(pos_items_list, dim=0).to(device)
        neg_items_tensor = torch.stack(neg_items_list, dim=0).to(device)
        return pos_items_tensor, neg_items_tensor

    def get_pos_label_tensor(self, pos_items, history_items):
        device = pos_items.device
        pos_items = pos_items.cpu().numpy().tolist()
        history_items = history_items.cpu().numpy().tolist()
        pos_label_tensor_list = []
        for p_item, h_item in zip(pos_items, history_items):
            p_item = set(p_item)
            p_item.discard(-1)
            if self.nbr_type == 'all':
                p_item_set = list(set(p_item))
            else:
                p_item_set = list(set(p_item)-set(h_item))
            pos_tensor = torch.zeros(self.n_items)
            pos_tensor[list(p_item)] = 1
            pos_label_tensor_list.append(pos_tensor)
        # print(pos_label_tensor_list)
        pos_label_tensor = torch.stack(pos_label_tensor_list).to(device)
        
        return pos_label_tensor
        
    
    def forward(self, item_seq, position_seq):
        # position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        # position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_seq)
        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        extended_attention_mask = self.get_attention_mask(item_seq, bidirectional=True)
        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        return output  # [B L H]

    def multi_hot_embed(self, masked_index, max_length):
        """
        For memory, we only need calculate loss for masked position.
        Generate a multi-hot vector to indicate the masked position for masked sequence, and then is used for
        gathering the masked position hidden representation.
        Examples:
            sequence: [1 2 3 4 5]
            masked_sequence: [1 mask 3 mask 5]
            masked_index: [1, 3]
            max_length: 5
            multi_hot_embed: [[0 1 0 0 0], [0 0 0 1 0]]
        """
        masked_index = masked_index.view(-1)
        multi_hot = torch.zeros(masked_index.size(0), max_length, device=masked_index.device)
        multi_hot[torch.arange(masked_index.size(0)), masked_index] = 1
        return multi_hot

    def calculate_bert_loss(self, item_seq, position_seq): # pos_items do not need.
        masked_item_seq, swaped_position_seq, pos_items, neg_items, masked_index = self.reconstruct_bert_train_data(item_seq, position_seq)

        seq_output = self.forward(masked_item_seq, swaped_position_seq)
        pred_index_map = self.multi_hot_embed(masked_index, masked_item_seq.size(-1))  # [B*mask_len max_len]
        # [B mask_len] -> [B mask_len max_len] multi hot
        pred_index_map = pred_index_map.view(masked_index.size(0), masked_index.size(1), -1)  # [B mask_len max_len]
        # [B mask_len max_len] * [B max_len H] -> [B mask_len H]
        # only calculate loss for masked position
        seq_output = torch.bmm(pred_index_map, seq_output)  # [B mask_len H]

        loss_fct = nn.CrossEntropyLoss(reduction='none') # we use CE loss here.
        test_item_emb = self.item_embedding.weight[:self.n_items]  # [item_num H]
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))  # [B mask_len item_num]
        targets = (masked_index > 0).float().view(-1)  # [B*mask_len]

        loss = torch.sum(loss_fct(logits.view(-1, test_item_emb.size(0)), pos_items.view(-1)) * targets) \
                / torch.sum(targets)
        return loss

    
    def calculate_nbr_loss(self, item_seq, position_seq): # pos_items do not need.
        masked_item_seq, swaped_position_seq, pos_items, neg_items, masked_index = self.reconstruct_nbr_train_data(item_seq, position_seq)

        seq_output = self.forward(masked_item_seq, swaped_position_seq)
        pred_index_map = self.multi_hot_embed(masked_index, masked_item_seq.size(-1))  # [B*mask_len max_len]
        # [B mask_len] -> [B mask_len max_len] multi hot
        pred_index_map = pred_index_map.view(masked_index.size(0), masked_index.size(1), -1)  # [B mask_len max_len]
        # [B mask_len max_len] * [B max_len H] -> [B mask_len H]
        # only calculate loss for masked position
        seq_output = torch.bmm(pred_index_map, seq_output)  # [B mask_len H]

        loss_fct = nn.CrossEntropyLoss(reduction='none') # we use CE loss here.
        test_item_emb = self.item_embedding.weight[:self.n_items]  # [item_num H]
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))  # [B mask_len item_num]
        targets = (masked_index > 0).float().view(-1)  # [B*mask_len]

        loss = torch.sum(loss_fct(logits.view(-1, test_item_emb.size(0)), pos_items.view(-1)) * targets) \
                / torch.sum(targets)
        return loss


    # def calculate_joint_loss(self, item_seq, position_seq, item_seq_len, pos_items, history_items, k=5):
    #     bert_loss = self.calculate_bert_loss
    def predict(self, item_seq, position_seq, item_seq_len, test_item):
        item_seq, position_seq = self.reconstruct_test_data(item_seq, position_seq, item_seq_len)
        seq_output = self.forward(item_seq, position_seq)
        seq_output = self.gather_indexes(seq_output, item_seq_len)  # [B H]
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, item_seq, position_seq, item_seq_len):
        item_seq, position_seq = self.reconstruct_test_data(item_seq, position_seq, item_seq_len)
        seq_output = self.forward(item_seq, position_seq)
        seq_output = self.gather_indexes(seq_output, item_seq_len)  # [B H] # not len-1 since we add an additional mask here, the output lenth is len+1
        test_items_emb = self.item_embedding.weight[:self.n_items]  # delete masked token
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, item_num]
        return scores

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def get_attention_mask(self, item_seq, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = (item_seq != 0)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1)))
        extended_attention_mask = torch.where(extended_attention_mask, 0., -10000.)
        return extended_attention_mask