"""Training script for the original model."""

from configs.data.dataset import UnlearningDataset
from configs.models.resnet import ResNetWrapper
from trainers.trainer import Trainer
from utils.checkpoint import save_model
from utils.config import load_config
from utils.seed import set_seed


def train():
    """
    Train the original model.

    Returns:
        Placeholder training result summary.
    """
    # TODO: Add script-level training orchestration.
    return {"status": "TODO", "stage": "train"}


def main():
    """
    Run the original model training pipeline.

    Steps:
    1. Load data
    2. Initialize model
    3. Train
    4. Save the original checkpoint
    """
    config = load_config("configs/config.yaml")
    set_seed(config.get("seed", 42))

    dataset = UnlearningDataset(config)
    model = ResNetWrapper(num_classes=config.get("num_classes", 10))
    trainer = Trainer(config)

    train_loader = dataset.get_retained_set()
    trainer.train(model, train_loader)
    save_model(model, "checkpoints/original.pt")
    return train()


if __name__ == "__main__":
    main()


import os
import torch
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from collections import defaultdict

from .consts import BENCHMARK, NON_BLOCKING
torch.backends.cudnn.benchmark = BENCHMARK                              

class GradualWarmupScheduler(_LRScheduler):
    """
    Gradually increase the learning rate during the warmup stage.

    Args:
        optimizer (Optimizer): Optimizer to update.
        multiplier (float): Learning-rate multiplier.
        total_epoch (int): Number of epochs used for warmup.
        after_scheduler (_LRScheduler): Scheduler applied after warmup.
    """
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        """Initialize the warmup scheduler."""
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater than or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        """Get the current learning rate."""
        if self.last_epoch > self.total_epoch:                
            if self.after_scheduler:
                if not self.finished:
                                  
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
                          
        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        """Step method for ReduceLROnPlateau."""
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1
        if self.last_epoch <= self.total_epoch:                
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        """Standard step method for non-ReduceLROnPlateau schedulers."""
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


def set_batchnorm_to_eval(module):
    for child in module.children():
        if isinstance(child, torch.nn.BatchNorm2d):
            child.eval()                      
        else:
            set_batchnorm_to_eval(child)

def train(model, loss_fn, trainloader, validloader, defs, model_class, setup=dict(dtype=torch.float, device=torch.device('cuda:0')),
          ckpt_path=None, finetune=False):
    """
    Main training entry point that trains a model using the provided settings.

    Args:
        model (torch.nn.Module): Model to train.
        loss_fn (callable): Loss function.
        trainloader (DataLoader): Training dataloader.
        validloader (DataLoader): Validation dataloader.
        defs (Namespace): Hyperparameter settings such as epochs and learning rate.
        setup (dict): Device and dtype configuration for training.
        ckpt_path (str): Checkpoint output path.
        finetune (bool): Whether the model is in finetuning mode.
    """
    stats = defaultdict(list)                   
    optimizer, scheduler = set_optimizer(model, defs)             

    for epoch in tqdm(range(1, defs.epochs+1)):                
        if finetune:
            model.eval()                
        else:
            model.train()            
        if model_class == 'resnet18':
            set_batchnorm_to_eval(model)


        step(model, loss_fn, trainloader, optimizer, scheduler, defs, setup, stats)          

        if epoch % defs.validate == 0 or epoch == defs.epochs:                   
            model.eval()        
            validate(model, loss_fn, validloader, defs, setup, stats)        
            print_status(epoch, loss_fn, optimizer, stats)          
            if ckpt_path is not None:                   
                torch.save(model.state_dict(), os.path.join(ckpt_path, f'model_{epoch}.pt'))

        if defs.dryrun:               
            break
        if not np.isfinite(stats['train_losses'][-1]):                         
            print('Loss became NaN/Inf. Stopping early...')
            break

    return stats                

def step(model, loss_fn, dataloader, optimizer, scheduler, defs, setup, stats):
    """
    Train the model for a single epoch.

    Args:
        model (torch.nn.Module): Model to train.
        loss_fn (callable): Loss function.
        dataloader (DataLoader): Dataloader.
        optimizer (Optimizer): Optimizer.
        scheduler (_LRScheduler): Learning-rate scheduler.
        defs (Namespace): Hyperparameter settings.
        setup (dict): Device and dtype settings.
        stats (dict): Storage for training statistics.
    """
    epoch_loss, epoch_metric = 0, 0                     
    for batch, (inputs, targets) in enumerate(dataloader):             
        optimizer.zero_grad()        
        inputs = inputs.to(**setup)                 
        targets = targets.to(device=setup['device'], non_blocking=NON_BLOCKING)          

        outputs = model(inputs)               
        loss, _, _ = loss_fn(outputs, targets)        

        epoch_loss += loss.item()        
        loss.backward()        
        optimizer.step()          

        metric, name, _ = loss_fn.metric(outputs, targets)               
        epoch_metric += metric.item()        

        if defs.scheduler == 'cyclic':                    
            scheduler.step()
        if defs.dryrun:              
            break
    if defs.scheduler == 'linear':                           
        scheduler.step()

    stats['train_losses'].append(epoch_loss / (batch + 1))          
    stats['train_' + name].append(epoch_metric / (batch + 1))          

def validate(model, loss_fn, dataloader, defs, setup, stats):
    """
    Evaluate model performance on the validation set.

    Args:
        model (torch.nn.Module): Model to evaluate.
        loss_fn (callable): Loss function.
        dataloader (DataLoader): Validation dataloader.
        defs (Namespace): Hyperparameter settings.
        setup (dict): Device and dtype settings.
        stats (dict): Storage for validation statistics.
    """
    epoch_loss, epoch_metric = 0, 0               
    with torch.no_grad():                  
        for batch, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(**setup)
            targets = targets.to(device=setup['device'], non_blocking=NON_BLOCKING)

            outputs = model(inputs)        
            loss, _, _ = loss_fn(outputs, targets)        
            metric, name, _ = loss_fn.metric(outputs, targets)        

            epoch_loss += loss.item()        
            epoch_metric += metric.item()        

            if defs.dryrun:              
                break

    stats['valid_losses'].append(epoch_loss / (batch + 1))               
    stats['valid_' + name].append(epoch_metric / (batch + 1))               

def set_optimizer(model, defs):
    """
    Configure the optimizer and learning-rate scheduler from defs.

    Args:
        model (torch.nn.Module): Model to optimize.
        defs (Namespace): Hyperparameter settings.
    """
    if defs.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=defs.lr, weight_decay=defs.weight_decay)
    elif defs.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=defs.lr, weight_decay=defs.weight_decay)

    if defs.scheduler == 'linear':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[100, 300,
                                                                     500], gamma=0.1)
                                       

    if defs.warmup:                   
        scheduler = GradualWarmupScheduler(optimizer, multiplier=10, total_epoch=10, after_scheduler=scheduler)

    return optimizer, scheduler

def print_status(epoch, loss_fn, optimizer, stats):
    """
    Print model status for the current epoch.

    Args:
        epoch (int): Current epoch.
        loss_fn (callable): Loss function.
        optimizer (Optimizer): Optimizer.
        stats (dict): Stored training and validation statistics.
    """
    current_lr = optimizer.param_groups[0]['lr']            
    name, format = loss_fn.metric()          
    print(f'Epoch: {epoch}| lr: {current_lr:.4f} | '
          f'Train loss is {stats["train_losses"][-1]:6.4f}, Train {name}: {stats["train_" + name][-1]:{format}} | '
          f'Val loss is {stats["valid_losses"][-1]:6.4f}, Val {name}: {stats["valid_" + name][-1]:{format}} |')           
