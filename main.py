"""
torchrun --nproc_per_node=4 --nnodes=1 --master_port 29600 main.py
python3 -m torch.distributed.run --nproc_per_node=2 --nnodes=1 --master_port 29602 main.py
"""


from datetime import timedelta
import wandb
from torch.distributed import init_process_group, destroy_process_group, barrier, all_reduce
import torch
import torch.nn as nn

from config import get_default_config
from common_utils import get_transformations, get_train_dataset, get_test_dataset, \
    get_test_dataloader, get_train_dataloader, get_optimizer, get_criterion, get_cuda_mem_stats
from model import get_model 
from train import Trainer

import os


if __name__=="__main__":
    config = get_default_config()
    
    config.local_rank = int(os.environ['LOCAL_RANK'])
    config.global_rank = int(os.environ['RANK'])
    config.world_size = int(os.environ['WORLD_SIZE'])
    
    init_process_group(backend="nccl", timeout=timedelta(seconds=7200000))
    torch.cuda.set_device(config.local_rank)
    
    # Setup wandb
    # if config.local_rank == 0:
    #     wandb.login()  
    
    # Setup model
    
    model = get_model(config)
    
    # Setup dataset
    train_transforms, val_transforms = get_transformations(config)
    train_dataset = get_train_dataset(config, train_transforms)
    test_dataset = get_test_dataset(config, val_transforms)
    train_dataloader, sampler = get_train_dataloader(config, train_dataset)
    test_dataloader = get_test_dataloader(config, test_dataset)
    
    # Setup optimizer
    scheduler, optimizer = get_optimizer(config, model, train_dataloader)
    
    model = model.cuda()
    model = nn.parallel.DistributedDataParallel(model, device_ids=[config.local_rank], output_device=config.local_rank, find_unused_parameters=True)
    
    # Setup criterion
    criterion = get_criterion(config).cuda()
    
    # Start training
    trainer = Trainer(
        config=config,
        model = model,
        optimizer = optimizer,
        scheduler = scheduler,
        criterion = criterion,
        train_dataloader = train_dataloader,
        test_dataloader = test_dataloader,
        sampler = sampler
        )
    trainer.train(config)
    trainer.save()
    
    destroy_process_group()
    
    