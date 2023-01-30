# Mask-Swap-NNBR
Source code for paper "Masked and Swapped Sequence Modeling for Next Novel Basket Recommendation in Grocery Shopping""

# Wandb

We use wandb to run and organize our results.

- Login to your wandb account, create a project and connect to it. See https://wandb.ai/quickstart for more details.
- Replace your project name in the following block in the train_main_gpu_batch.py
```
wandb.init(project="YOUR PROJECT", 
        name=f"{config['dataset']}_nbr{config['loss_type']}{config['nbr_type']}{config['bask_aug']}{config['nbr_loss']}_bert{config['mask_type']}{config['bert_loss']}_{config['swap_hop']}swap{config['swap_ratio']}_mask{config['mask_ratio']}_cat{config['cat_info']}_{config['foldk']}",
        config=config)
```
- Method 1: using the following script to config and run:
```
python train_main_gpu_batch.py --batch_size=64 --bert_loss=1 --dataset=tafeng --dropout_prob=0.1 --epochs=50 --foldk=0 --hidden_size=128 --mask_ratio=0.1 --mask_type=select --nbr_loss=2 --nbr_type=all --swap_hop=1 --swap_ratio=0
```
- Method 2: using wandb sweep to run multiple versions on slurm:

#### An example yaml to create a sweep job in wandb

```
method: grid
metric:
  name: best_test_recall10
parameters:
  batch_size:
    value: 64
  bert_loss:
    value: 1
  dataset:
    value: dunnhumby
  dropout_prob:
    value: 0.1
  epochs:
    value: 50
  foldk:
    values: [0, 1, 2, 3, 4 ]
  hidden_size:
    value: 128
  mask_ratio:
    value: 0.5
  mask_type:
    value: select
  nbr_loss:
    value: 2
  nbr_type:
    values:
      - explore
      - all
  swap_hop:
    value: 1
  swap_ratio:
    value: 0
program: train_main_gpu_batch.py
```
After you create the sweep task, you can start the sweep agent on slurm to run different versions in parallel, see https://docs.wandb.ai/guides/sweeps/start-sweep-agents for more details.

- The results will be shown in your wandb project automatically!! 

- Note that, you do not want to use wandb to run our code, just replace the wandb part in our code. It should be easy.
  





