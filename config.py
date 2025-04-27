from dataclasses import dataclass, field
import os
from cityscapes_config import Config


DATASET = "Cityscapes3D"

@dataclass
class PascalConfig:
    # image config
    image_size: int = (512, 512)
    embed_dim: int = 96
    dataset: str = "PASCAL_MT"
    
    # decoder config
    decoder_depth: int = 2
    decoder_n_head: list = field(default_factory=lambda: [48, 24, 12, 6])  
    decoder_window_size: list = field(default_factory=lambda: [4, 4, 8, 8])
    distillation = False
    model_name = "distillation-edge-normals"
    
    # dataset
    project_path: str = os.path.dirname(os.path.abspath(__file__))
    pascal_path: str = os.path.join(project_path, "PASCAL_MT")
    train_batch_size: int = 8
    val_batch_size: int = 32
    num_workers_data: int = 2
    
    
    # task config
    tasks = {
        "edge" : False, 
        "human_parts" : False, 
        "semseg" : True, 
        "normals" : False, 
        "sal" : False
        }
    ignore_index: int = 255
    loss_weights = {
        "semseg": 1.0,
        # "human_parts": 2.0,
        # "sal": 5.0,
        # "edge": 50.0,
        # "normals": 10.0,
    }
    num_classes = {
        "semseg": 21,
        "human_parts": 7,
        "sal": 2,
        "edge": 1,
        "normals": 3
    }
       
    
    # Optimizer
    epochs: int = 70
    warmup_steps: int =1000
    optimizer_kwargs = {
        "lr": 0.00005,
        "weight_decay": 0.000001
    }
    val_interval: int = 500
    save_interval: int = 500
    resume_training: bool = False
    
    # wandb
    wandb_project: str = "encoder-decoder-pascal-single"
    wandb_run_name: str = "semseg-single-encoder-decoder"

@dataclass
class CityscapesConfig:
    # image config
    image_size = [512, 1024]
    embed_dim: int = 96
    dataset: str = "Cityscapes3D"
    
    # decoder config
    decoder_depth: int = 2
    decoder_n_head: list = field(default_factory=lambda: [48, 24, 12, 6])  
    decoder_window_size: list = field(default_factory=lambda: [4, 4, 8, 8])
    distillation = True
    model_name = "distillation-edge-normals"
    
    # dataset
    project_path: str = os.path.dirname(os.path.abspath(__file__))
    pascal_path: str = os.path.join(project_path, "Cityscapes3D")
    train_batch_size: int = 1
    val_batch_size: int = 1
    num_workers_data: int = 1
    
    
    # task config
    tasks = {
        "depth" : True, 
        "semseg" : True, 
        }
    ignore_index: int = 255
    loss_weights = {
        "semseg": 13.0,
        "depth": 1.0
        # "semseg": 1.0,
        # "depth": 1.0
    }
    num_classes = {
        "semseg": 19,
        "depth": 1
    }
    label_map_ratio = 1
       
    
    # Optimizer
    epochs: int = 20
    warmup_steps: int =1000
    optimizer_kwargs = {
        "lr": 0.00005,
        "weight_decay": 0.000001
    }
    val_interval: int = 1000
    save_interval: int = 1000
    resume_training: bool = True
    
    # wandb
    wandb_project: str = "encoder-decoder-cityscapes-single"
    wandb_run_name: str = "cityscapes-single-depth"
    
def get_default_config():
    if DATASET=="Cityscapes3D":
        return CityscapesConfig()
    else:
        return PascalConfig()
    
    