import os

from common_utils import to_cuda, get_output, get_cuda_mem_stats
from evaluation.eval_utils import PerformanceMeter

import torch
from tqdm import tqdm
import wandb



class Trainer:
    def __init__(self, config, model, optimizer, scheduler, criterion, train_dataloader, test_dataloader, sampler, model_name = "encoder-decoder") -> None:
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.sampler = sampler
        
        self.model_name = config.model_name
        
        self.iter_count = 0
        self.n_epochs = 0
        self.results_dir = os.path.join(self.config.project_path, "results")
        
        self.tasks = []
        for task, include_task in config.tasks.items():
            if include_task:
                self.tasks.append(task)

    def train_step(self, batch):
        batch = to_cuda(batch)
        images = batch["image"]
        output = self.model.module(images)
        
        loss_dict = self.criterion(output, batch)
        
        self.optimizer.zero_grad()
        loss_dict["total"].backward()
        self.optimizer.step()
        self.scheduler.step()
        
        if self.config.local_rank==0:
            wandb.log({"train_loss": loss_dict["total"].item()}, step=self.iter_count)
            for task in self.tasks:
                wandb.log({f"train_{task}_loss": loss_dict[task].item()}, step=self.iter_count)
                if self.config.distillation:
                    wandb.log({f"train_initial_{task}_loss": loss_dict[f"initial_{task}"].item()}, step=self.iter_count)
            wandb.log({"learning rate": self.scheduler.get_last_lr()[0]}, step=self.iter_count)
        
        return loss_dict["total"].item(), loss_dict
    
    def log_eval_results(self, eval_results):
        for task in self.tasks:
            for metric, value in eval_results[task].items():
                wandb.log({f"val_{task}_{metric}": value}, step=self.iter_count)
    
    def val_step(self):
        print("Validation")
        performence_meter = PerformanceMeter(self.config, self.tasks)
        self.model.eval()
        epoch_loss = 0
        val_loss_dict = {t: 0 for t in self.tasks}
        print("Doing Validation")
        with torch.no_grad():
            pbar = tqdm(self.test_dataloader, disable=self.config.local_rank != 0)
            for i, batch in enumerate(pbar):
                batch = to_cuda(batch)
                images = batch["image"]
                targets = {task: batch[task] for task in self.tasks}
                output = self.model.module(images)
                loss_dict = self.criterion(output, batch)
                epoch_loss += loss_dict["total"].item()
                for t in self.tasks:
                    val_loss_dict[t] += loss_dict[t].item()
                    
                performence_meter.update(
                    {t: get_output(output[t], t) for t in self.tasks},
                    {t: targets[t] for t in self.tasks}
                ) 
                # if i==20:
                #     break
        
        wandb.log({"val_loss": epoch_loss/len(self.test_dataloader)}, step=self.iter_count)
        for task in self.tasks:
            wandb.log({f"val_{task}_loss": val_loss_dict[task]/len(self.test_dataloader)}, step=self.iter_count)
        
        eval_results = performence_meter.get_score(verbose=True)
        self.log_eval_results(eval_results)
        print("Validation Done")
        print()
        return epoch_loss/len(self.test_dataloader)
    
    def train(self, config):
        if config.local_rank == 0:
            wandb.init(project=config.wandb_project, name=config.wandb_run_name, config=self.config.__dict__)
        
        self.model.train()
        for epoch in range(self.n_epochs, self.config.epochs):
            self.n_epochs = epoch
            epoch_loss = 0
            self.sampler.set_epoch(epoch)
            pbar = tqdm(self.train_dataloader, disable=self.config.local_rank != 0)
            torch.distributed.barrier()
            for i, batch in enumerate(pbar):
                loss, loss_dict = self.train_step(batch)
                epoch_loss+=loss
                
                pbar.set_description(f"Epoch {epoch}, Loss: {loss:.4f}, step: {i}")
                
                if config.local_rank==0 and self.iter_count%config.val_interval==0:
                    val_loss = self.val_step()
                if config.local_rank==0 and self.iter_count%config.save_interval==0:
                    self.save()
                torch.distributed.barrier()
                
                if config.local_rank==0 and self.iter_count%config.val_interval==0:
                    print(f"Epoch: {epoch}| Iteration: {self.iter_count}| TrainingLoss: {loss}| ValidationLoss: {val_loss}")
                    print()
                
                self.iter_count += 1
            
            torch.distributed.barrier()
            epoch_loss /= len(self.train_dataloader)
            if config.local_rank==0:
                wandb.log({"epoch_loss_train": epoch_loss}, step=self.iter_count)
            torch.distributed.barrier()
        
    def load(self):
        checkpoint_path = os.path.join(self.results_dir, f"checkpoint-{self.model_name}-{self.config.dataset}")
        if self.config.resume_training and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint["model"])
            self.model = self.model.cuda()
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.iter_count = checkpoint["iter_count"]
            self.n_epochs = checkpoint["epoch"]
            print(f"Loaded checkpoint from {checkpoint_path}")
            
        
    def save(self):
        checkpoint_dict = {
            'model': self.model.state_dict(), 
            'optimizer': self.optimizer.state_dict(), 
            'scheduler': self.scheduler.state_dict(), 
            'iter_count': self.iter_count+1,
            'epoch': self.n_epochs+1,
            # "wandb_run_id": wandb.run.id
        }
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        results_path = os.path.join(self.results_dir, f"checkpoint-{self.model_name}-{self.config.dataset}")
        torch.save(checkpoint_dict, results_path)
        
        
        