from data import transforms
from data.pascal_context import PASCALContext
from data.cityscapes3d import CITYSCAPES3D
from losses.loss_functions import *
from losses.multitask_loss import MultiTaskLoss, MultiTaskDistillationLoss

import torchvision
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
from transformers import get_cosine_schedule_with_warmup

def get_transformations(config):
    if config.dataset=="PASCAL_MT":
        train_transforms = torchvision.transforms.Compose([ # from ATRC
            transforms.RandomScaling(scale_factors=[0.5, 2.0], discrete=False),
            transforms.RandomCrop(size=(512,512), cat_max_ratio=0.75),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.PhotoMetricDistortion(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.PadImage(size=(512,512)),
            transforms.AddIgnoreRegions(),
            transforms.ToTensor(),
        ])

        # Testing 
        valid_transforms = torchvision.transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.PadImage(size=(512,512)),
            transforms.AddIgnoreRegions(),
            transforms.ToTensor(),
        ])
        return train_transforms, valid_transforms
    else:
        return None, None

def get_train_dataset(config, transforms):
    tasks = set()
    for task, include_task in config.tasks.items():
        if include_task:
            tasks.add(task)
    
    if config.dataset=="PASCAL_MT":
        train_dataset = PASCALContext(
            config.pascal_path, 
            download=False, 
            split=['train'], 
            transform=transforms, 
            retname=True,
            do_semseg="semseg" in tasks,
            do_edge="edge" in tasks,
            do_normals="normals" in tasks,
            do_sal="sal" in tasks,
            do_human_parts="human_parts" in tasks,
            overfit=False
            )
    else:
        train_dataset = CITYSCAPES3D(
            config,
            root="Cityscapes3D",
            split=["train", "val"],
            img_size=config.image_size,
            task_list=list(tasks)
        )
    return train_dataset

def get_test_dataset(config, transforms):
    tasks = set()
    for task, include_task in config.tasks.items():
        if include_task:
            tasks.add(task)
    
    if config.dataset=="PASCAL_MT":
        test_dataset = PASCALContext(
            config.pascal_path, 
            download=False, 
            split=['val'], 
            transform=transforms, 
            retname=True,
            do_semseg="semseg" in tasks,
            do_edge="edge" in tasks,
            do_normals="normals" in tasks,
            do_sal="sal" in tasks,
            do_human_parts="human_parts" in tasks,
            overfit=False
            )
    else:
        test_dataset = CITYSCAPES3D(
            config,
            root="Cityscapes3D",
            split=["test"],
            img_size=config.image_size,
            task_list=list(tasks)
        )
    return test_dataset

def get_train_dataloader(config, dataset, infer=False):
    if infer:
        return DataLoader(
            dataset, 
            batch_size=config.train_batch_size, 
            shuffle = False,
            num_workers=config.num_workers_data, 
            pin_memory=False, 
            )
    train_sampler = DistributedSampler(dataset, shuffle=True)
    trainloader = DataLoader(
        dataset, 
        batch_size=config.train_batch_size, 
        shuffle = False,
        num_workers=config.num_workers_data, 
        pin_memory=False, 
        sampler=train_sampler
        )
    return trainloader, train_sampler

def get_test_dataloader(config, dataset):
    testloader = DataLoader(
        dataset, 
        batch_size=config.val_batch_size, 
        shuffle = True,
        num_workers=config.num_workers_data, 
        pin_memory=False
        )
    return testloader


def get_optimizer(config, model, dataloader):
    
    params = model.parameters()

    optimizer = torch.optim.Adam(params, **config.optimizer_kwargs)
    
    # get scheduler
    num_training_steps = len(dataloader) * config.epochs  # Adjust based on your data and epochs
    num_warmup_steps = config.warmup_steps 

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps
    )
    
    return scheduler, optimizer


def get_loss(config, task=None):
    """ Return loss function for a specific task """

    if task == 'edge':
        criterion = BalancedBinaryCrossEntropyLoss(pos_weight=0.95, ignore_index=config.ignore_index)

    elif task == 'semseg' or task == 'human_parts':
        criterion = CrossEntropyLoss(ignore_index=config.ignore_index)

    elif task == 'normals':
        criterion = L1Loss(normalize=True, ignore_index=config.ignore_index)

    elif task == 'sal':
        criterion = CrossEntropyLoss(balanced=True, ignore_index=config.ignore_index) 

    elif task == 'depth':
        criterion = L1Loss(ignore_invalid_area=True, ignore_index=-1)

    else:
        criterion = None

    return criterion

def get_criterion(config):
    loss_function = torch.nn.ModuleDict()
    tasks = []
    
    for task, include_task in config.tasks.items():
        if include_task:
            tasks.append(task)
            loss_function[task] = get_loss(config, task)
    
    if config.distillation:
        return MultiTaskDistillationLoss(tasks, loss_function, config.loss_weights)
    else:
        return MultiTaskLoss(config, tasks, loss_function, config.loss_weights)
    

def to_cuda(batch):
    if type(batch) == dict:
        out = {}
        for k, v in batch.items():
            if k == 'meta':
                out[k] = v
            else:
                out[k] = to_cuda(v)
        return out
    elif type(batch) == torch.Tensor:
        return batch.cuda()
    elif type(batch) == list:
        return [to_cuda(v) for v in batch]
    else:
        return batch
    
def get_cuda_mem_stats(config):
    return torch.cuda.memory_allocated(f"cuda:{config.local_rank}")/(1024**3)
    
def get_output(output, task):
    
    if task == 'normals':
        output = output.permute(0, 2, 3, 1)
        output = (F.normalize(output, p = 2, dim = 3) + 1.0) * 255 / 2.0
    
    elif task == 'semseg':
        output = output.permute(0, 2, 3, 1)
        _, output = torch.max(output, dim=3)

    elif task == 'human_parts':
        output = output.permute(0, 2, 3, 1)
        _, output = torch.max(output, dim=3)
    
    elif task == 'edge':
        output = output.permute(0, 2, 3, 1)
        output = torch.squeeze(255 * 1 / (1 + torch.exp(-output)))

    elif task == 'sal':
        output = output.permute(0, 2, 3, 1)
        output = F.softmax(output, dim=3)[:, :, :, 1] *255 
    
    elif task=="depth":
        output.clamp_(min=0.)
        output = output.permute(0, 2, 3, 1)
    
    else:
        raise ValueError('Select one of the valid tasks')

    return output

