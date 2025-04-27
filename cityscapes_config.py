from dataclasses import dataclass, field
import os

@dataclass
class Config:
    # image config
    image_size: int = (1024, 2048)
    embed_dim: int = 96
    dataset: str = "Cityscapes3D"
    
    # decoder config
    decoder_depth: int = 2
    decoder_n_head: list = field(default_factory=lambda: [48, 24, 12, 6])  
    decoder_window_size: list = field(default_factory=lambda: [4, 4, 8, 8])
    
    # dataset
    project_path: str = os.path.dirname(os.path.abspath(__file__))
    pascal_path: str = os.path.join(project_path, "Cityscapes3D")
    train_batch_size: int = 8
    val_batch_size: int = 32
    num_workers_data: int = 2
    
    
    # task config
    tasks = {
        "depth" : True, 
        "semseg" : True, 
        }
    ignore_index: int = 255
    loss_weights = {
        "semseg": 100.0,
        "depth": 1.0
    }
    num_classes = {
        "semseg": 19,
        "depth": 1
    }
       
    
    # Optimizer
    epochs: int = 50
    warmup_steps: int =1000
    optimizer_kwargs = {
        "lr": 0.00005,
        "weight_decay": 0.000001
    }
    val_interval: int = 500
    save_interval: int = 500
    resume_training: bool = False
    
    # wandb
    wandb_project: str = "encoder-decoder-dlcv-cityscapes"
    wandb_run_name: str = "test-1"
    
def get_default_config():
    return Config()
    
    