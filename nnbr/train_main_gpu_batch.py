import os
from re import X
import sys
from logging import getLogger
from time import time
import datetime
import argparse
import yaml
import pickle
import json

import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
# from tqdm import tqdm

# import models
from metrics import *
from bert4nbr import BERT4Expl
from dataloader import SeqDataset, get_dataloader
from utils import convert_to_gpu, convert_all_data_to_gpu
import itertools

import wandb


# import debugpy
# debugpy.listen(("0.0.0.0", 12416))
# import metrics

def get_config(config_path=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type = str)
    parser.add_argument('--foldk', type=int, default=0)
    parser.add_argument('--mask_type', type=str, default='select')
    parser.add_argument('--mask_ratio', type=float, default=0.2)
    parser.add_argument('--bask_mask_ratio', type=float, default=1.0)
    parser.add_argument('--bask_aug', type=int, default=0)
    parser.add_argument('--dropout_prob', type = float, default=0.1)
    parser.add_argument('--swap_ratio', type=float, default=0, help='determine how much repeat samples we will use during training.')
    parser.add_argument('--swap_hop', type=int, default=1, help='move item to more distances')
    parser.add_argument('--max_seq_length', type=int, default=256)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--method', type=str, default='bert4rec')
    parser.add_argument('--loss_type', type=str, default='BPR', help='BPR or BCE or RE')
    parser.add_argument('--nbr_type', type=str, default='all', help='all or explore')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--cat_info', type=int, default=0, help='use cat info to predict 1, Never used this in this project.')
    parser.add_argument('--nbr_loss', type=int, default=0)
    parser.add_argument('--bert_loss', type=int, default=0)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--n_heads', type=int, default=8)

    args, _ = parser.parse_known_args()
    input_config = args.__dict__
    if input_config['dataset'] == 'tafeng':
        dataset_config = {'n_items': 11998, 'n_users': 13859}
    elif input_config['dataset'] == 'dunnhumby':
        dataset_config = {'n_items': 3921, 'n_users': 22531}
    elif input_config['dataset'] == 'instacart':
        dataset_config = {'n_items': 13898, 'n_users': 19436}

    model_config_path = f"model_config.yaml"
    with open(model_config_path, 'rb') as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)
    config = {**dataset_config, **model_config, **input_config}

    config['foldk_path'] = f"../keyset/{input_config['dataset']}_keyset_{input_config['foldk']}.json"
    config['data_path'] = f"../mergeddata/{input_config['dataset']}_merged.json"
    # config['cat_info_path'] = f"../cat_info/{config['dataset']}item_cat_info.pkl" ## this is never used in this project.
    # config['model_folder'] = f"model/{config['dataset']}/{config['mask_type']}/swap_{config['swap_ratio']}_mask_{config['mask_ratio']}/{config['foldk']}/"
    # config['eval_folder'] = f"eval/{config['dataset']}/{config['mask_type']}/swap_{config['swap_ratio']}_mask_{config['mask_ratio']}/{config['foldk']}/"
    config['prediction_folder'] = f"pred/{config['dataset']}/nbr{config['loss_type']}{config['nbr_type']}{config['bask_aug']}{config['nbr_loss']}_bert{config['mask_type']}{config['bert_loss']}_swap{config['swap_ratio']}_mask{config['mask_ratio']}_cat{config['cat_info']}/{config['foldk']}/"
    # if not os.path.exists(config['model_folder']):
    #     os.makedirs(config['model_folder'])
    # if not os.path.exists(config['eval_folder']):
    #     os.makedirs(config['eval_folder'])
    if not os.path.exists(config['prediction_folder']):
        os.makedirs(config['prediction_folder'])

    if torch.cuda.is_available():
        print("CUDA!!")
        config['device'] = torch.device('cuda')
    else:
        config['device'] = torch.device('cpu')
        print("CPU!!")
    # some require config['device']
    sys.stdout.flush()
    
    return config

def eval_acc_expl(topk_list, user_list, hist_item_list, predict_list, label_list, user_cand_dict, config):
    # 
    acc_dict = {acc+str(topk):[] for acc in ['recall', 'ndcg'] for topk in topk_list}
    acc_dict['user'] = []
    expl_pred_dict = dict()
    for user, hist_item, predict, label in zip(user_list, hist_item_list, predict_list, label_list):
        expl_label = set(label)-set(hist_item)
        expl_label.discard(-1) 
        expl_label.discard(0)
        if len(expl_label) == 0: # if no expl_label, then drop this user
            continue
        acc_dict['user'].append(user)
        
        # hist_cat = [cat_info]
        if config['cat_info'] == 0:
            predict = [item for item in predict if item not in set(hist_item)] #exploration prediction
        else:
            cand_item = user_cand_dict[user]
            if len(cand_item) == 0:
                predict = [item for item in predict if item not in set(hist_item)] #exploration prediction
            else:
                predict = [item for item in predict if (item not in set(hist_item)) and (item in cand_item)] #exploration prediction
        expl_pred_dict[user] = predict[:50]
        for topk in topk_list:
            acc_dict['recall'+str(topk)].append(get_Recall(expl_label, predict, topk))
            acc_dict['ndcg'+str(topk)].append(get_NDCG(expl_label, predict, topk))
    avg_acc_dict = {metric: np.mean(result) for metric, result in acc_dict.items() if metric not in ['user', 'rep_user', 'expl_user']}
    return avg_acc_dict, expl_pred_dict
    
def check_nan(loss):
    if torch.isnan(loss):
        raise ValueError('Training loss is nan')

def create_model(config):
    if config['method'] == 'bert4rec':
        model = BERT4Expl(config)
    return model

def eval_model(model, dataloader, user_cand_dict, type, config):
    topk_list = [1, 5, 10, 20]

    model.eval()
    rec_ranks_list = []
    user_list = []
    pos_item_list = []
    item_hist_list = []
    for step, (user, item_seq, position_seq, item_seq_len, pos_items, history_items) in enumerate(dataloader):
        item_seq, position_seq, item_seq_len, pos_items, history_items  = convert_all_data_to_gpu(item_seq, position_seq, item_seq_len, pos_items, history_items, device=config['device'])
        predict = model.full_sort_predict(item_seq, position_seq, item_seq_len) # [b, n_items]
        predict = predict.cpu()
        _, rec_ranks = torch.topk(predict, 500, dim=-1) # save top 100
        rec_ranks_list.extend(rec_ranks.tolist())
        pos_item_list.extend(pos_items.cpu().numpy().tolist())
        user_list.extend(user.data.tolist())
        item_hist_list.extend(history_items.cpu().numpy().tolist())

    acc_dict, expl_pred_dict = eval_acc_expl(topk_list, user_list, item_hist_list, rec_ranks_list, pos_item_list, user_cand_dict, config)
    wandb_acc_dict = {}
    for m, v in acc_dict.items():
        wandb_acc_dict[f'{type}@{m}'] = v
    wandb.log(wandb_acc_dict)
    return acc_dict, user_list, expl_pred_dict

def check_update_model(val_acc_dict, test_acc_dict, val_best_acc_dict, test_best_acc_dict, test_expl_pred_dict):
    val_metric = val_acc_dict['recall5'] + val_acc_dict['recall10'] + val_acc_dict['recall20']
    val_best_metric = val_best_acc_dict['recall5'] + val_best_acc_dict['recall10'] + val_best_acc_dict['recall20']

    if val_metric > val_best_metric:
        print('====find better model based on val!====') 
        val_best_acc_dict = val_acc_dict
        test_best_acc_dict = test_acc_dict
    
        # # save prediction eval model
        # predict_dict = {usr: rec_list for usr, rec_list in zip(test_user_list, test_rec_ranks_list)}
        with open(config['prediction_folder']+"best_overall_prediction.json", 'w') as f:
            json.dump(test_expl_pred_dict, f)
        # acc_dict = {'best_epoch': epoch, 'overall': test_acc_dict}
        # with open(config['eval_folder']+"best_overall_acc.json", 'w') as f:
        #     json.dump(acc_dict, f)
        # torch.save(model.state_dict(), config['model_folder']+"best_overall_acc.pkl")
    
    metric_list = ['recall5', 'recall10', 'recall20', 'ndcg5', 'ndcg10', 'ndcg20']
    wandb_dict = dict()
    for metric in metric_list:
        wandb_dict[f'best_val_{metric}'] = val_best_acc_dict[metric]
        wandb_dict[f'best_test_{metric}'] = test_best_acc_dict[metric]
    wandb.log(wandb_dict)
    return val_best_acc_dict, test_best_acc_dict


def train_model(model:BERT4Expl, optimizer, optimizer_2, train_bert_data_loader, train_nbr_data_loader, val_data_loader, test_data_loader, config):
    print(model)
    print(optimizer)

    model = convert_to_gpu(model, device=config['device'])
    model.train()
    start_time = datetime.datetime.now()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer) # default one
    scheduler_2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_2) # two stage optimzer

    user_cand_dict = None
    # print(user_cand_dict.keys())
    topk_list = [1, 5, 10, 20]
    # bert_epoch = config['bert_epochs']
    
    val_best_acc_dict = {acc+str(topk): 0 for acc in ['recall', 'ndcg', 'mrr'] for topk in topk_list}
    test_best_acc_dict = {acc+str(topk): 0 for acc in ['recall', 'ndcg', 'mrr'] for topk in topk_list}
    val_metric_anchor = 'recall10'
    # test_acc_all_dict = dict()

    for epoch in range(config['epochs']):

        bert_total_loss = 0
        nbr_total_loss = 0
        mix_total_loss = 0
        if (config['nbr_loss'] in [1, 2]) and (config['bert_loss'] == 1): #joint train only use the bert data loader
            for step, (user, item_seq, position_seq, item_seq_len, pos_items, history_items) in enumerate(train_nbr_data_loader):
                model.train()
                item_seq, position_seq, item_seq_len, pos_items, history_items  = convert_all_data_to_gpu(item_seq, position_seq, item_seq_len, pos_items, history_items, device=config['device'])
                # nbr loss
                # if epoch>(config['epochs']/2.0):
                ################
                # optimizer.zero_grad()
                # nbr_loss = model.calculate_nbr_loss(item_seq, position_seq)
                # nbr_total_loss += nbr_loss.cpu().data.numpy()
                # check_nan(nbr_loss)
                # nbr_loss.backward()
                # optimizer.step()
                # # else:
                # # bert loss
                # optimizer.zero_grad()
                # bert_loss = model.calculate_bert_loss(item_seq, position_seq)
                # bert_total_loss += bert_loss.cpu().data.numpy()
                # check_nan(bert_loss)
                # bert_loss.backward()
                # optimizer.step()
                if config['nbr_loss'] == 1:
                    optimizer.zero_grad()
                    nbr_loss = model.calculate_nbr_loss(item_seq, position_seq)
                    check_nan(nbr_loss)
                    bert_loss = model.calculate_bert_loss(item_seq, position_seq)
                    check_nan(bert_loss)
                    mix_loss = (nbr_loss+bert_loss)/2
                    mix_total_loss += mix_loss.cpu().data.numpy()
                    mix_loss.backward()
                    optimizer.step()
                else:
                    if epoch > int(config['epochs']/2.0):
                        optimizer_2.zero_grad()
                        nbr_loss = model.calculate_nbr_loss(item_seq, position_seq)
                        nbr_total_loss += nbr_loss.cpu().data.numpy()
                        check_nan(nbr_loss)
                        nbr_loss.backward()
                        optimizer_2.step()
                    else:
                    # bert loss
                        optimizer.zero_grad()
                        bert_loss = model.calculate_bert_loss(item_seq, position_seq)
                        bert_total_loss += bert_loss.cpu().data.numpy()
                        check_nan(bert_loss)
                        bert_loss.backward()
                        optimizer.step()


                if step%20 == 0:
                    val_acc_dict, val_user_list, val_expl_pred_dict = eval_model(model, val_data_loader, user_cand_dict, 'val', config)
                    test_acc_dict, test_user_list, test_expl_pred_dict = eval_model(model, test_data_loader, user_cand_dict, 'test', config)
                    val_best_acc_dict, test_best_acc_dict = check_update_model(val_acc_dict, test_acc_dict, val_best_acc_dict, test_best_acc_dict, test_expl_pred_dict)

            epoch_avg_bertloss = bert_total_loss/(step+1)
            epoch_avg_nbrloss = nbr_total_loss/(step+1)
            epoch_avg_loss = mix_total_loss/(step+1)
            # wandb.log({'joint_loss': epoch_avg_loss}, step=epoch+1)

        elif (config['nbr_loss'] == 0) and (config['bert_loss'] == 1):
            for step, (user, item_seq, position_seq, item_seq_len, pos_items, history_items) in enumerate(train_bert_data_loader):
                model.train()
                item_seq, position_seq, item_seq_len, pos_items, history_items  = convert_all_data_to_gpu(item_seq, position_seq, item_seq_len, pos_items, history_items, device=config['device'])
                optimizer.zero_grad()
                loss = model.calculate_bert_loss(item_seq, position_seq)
                bert_total_loss += loss.cpu().data.numpy()
                check_nan(loss)
                loss.backward()
                optimizer.step()

                if step%20 == 0:
                    val_acc_dict, val_user_list, val_expl_pred_dict = eval_model(model, val_data_loader, user_cand_dict, 'val', config)
                    test_acc_dict, test_user_list, test_expl_pred_dict = eval_model(model, test_data_loader, user_cand_dict, 'test', config)
                    val_best_acc_dict, test_best_acc_dict = check_update_model(val_acc_dict, test_acc_dict, val_best_acc_dict, test_best_acc_dict, test_expl_pred_dict)

            epoch_avg_bertloss = bert_total_loss/(step+1)
            wandb.log({'bert_loss': epoch_avg_bertloss}, step=epoch+1)

        elif (config['nbr_loss'] == 1) and (config['bert_loss'] == 0):
            for step, (user, item_seq, position_seq, item_seq_len, pos_items, history_items) in enumerate(train_nbr_data_loader):
                model.train()
                item_seq, position_seq, item_seq_len, pos_items, history_items  = convert_all_data_to_gpu(item_seq, position_seq, item_seq_len, pos_items, history_items, device=config['device'])
                optimizer.zero_grad()
                loss = model.calculate_nbr_loss(item_seq, position_seq)
                nbr_total_loss += loss.cpu().data.numpy()
                check_nan(loss)
                loss.backward()
                optimizer.step()
            
                if step%20 == 0:
                    val_acc_dict, val_user_list, val_expl_pred_dict = eval_model(model, val_data_loader, user_cand_dict, 'val', config)
                    test_acc_dict, test_user_list, test_expl_pred_dict = eval_model(model, test_data_loader, user_cand_dict, 'test', config)
                    val_best_acc_dict, test_best_acc_dict = check_update_model(val_acc_dict, test_acc_dict, val_best_acc_dict, test_best_acc_dict, test_expl_pred_dict)

            epoch_avg_nbrloss = nbr_total_loss/(step+1)
            wandb.log({'nbr_loss': epoch_avg_nbrloss}, step=epoch+1)
        
        
        # loss scheduler
        if (config['nbr_loss'] in [1, 2]) and (config['bert_loss'] == 1):
            if config['nbr_loss'] == 1:
                scheduler.step(epoch_avg_loss)
            elif config['nbr_loss'] == 2:
                if epoch>int(config['epochs']/2):
                    scheduler_2.step(epoch_avg_nbrloss)
                else:
                    scheduler.step(epoch_avg_bertloss)

        elif (config['nbr_loss'] == 1) and (config['bert_loss'] == 0):
            scheduler.step(epoch_avg_nbrloss)
        elif (config['nbr_loss'] == 0) and (config['bert_loss'] == 1):
            scheduler.step(epoch_avg_bertloss)

    # with open(config['eval_folder']+"test_acc_every_epoch", 'w') as f:
    #     json.dump(test_acc_all_dict, f)
    end_time = datetime.datetime.now()
    print(f'End! Total cost {end_time-start_time}!')    
                    
if __name__ == '__main__':

    config = get_config()
    # replace your project name here.
    wandb.init(project="YOUR PROJECT", 
                name=f"{config['dataset']}_nbr{config['loss_type']}{config['nbr_type']}{config['bask_aug']}{config['nbr_loss']}_bert{config['mask_type']}{config['bert_loss']}_{config['swap_hop']}swap{config['swap_ratio']}_mask{config['mask_ratio']}_cat{config['cat_info']}_{config['foldk']}",
                config=config)
    config = wandb.config
    model = create_model(config)
    os.environ['CUDA_LAUNCH_BLOCKING']= '1'


    train_bert_data_loader, train_nbr_data_loader, val_data_loader, test_data_loader = get_dataloader(config, d_type='train'), get_dataloader(config, d_type='train-nbr'), get_dataloader(config, d_type='val'), get_dataloader(config, d_type='test')
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'])
    optimizer_2 = torch.optim.Adam(model.parameters(),
                                lr=config['learning_rate'],
                                weight_decay=config['weight_decay'])
    train_model(model, optimizer, optimizer_2, train_bert_data_loader, train_nbr_data_loader, val_data_loader, test_data_loader, config)
    