import torch
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm
from shutil import copyfile
from torch.optim.lr_scheduler import LambdaLR
import os


class BaseTask:
    def __init__(self, config, model, train_dataloader, dev_dataloader):
        super().__init__()
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(params=self.model.parameters(),
                                    lr=config['TRAINING']['LEARNING_RATE'],
                                    betas=(0.9, 0.98))
        
        self.epoch = config['TRAINING']['EPOCH']
        self.running_epoch = 0
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        
        self.patience = config['TRAINING']['PATIENCE']
        self.device = config['TRAINING']['DEVICE']
        self.score = config['TRAINING']['SCORE']
        self.warmup = config['TRAINING']['WARMUP']
        self.scheduler = LambdaLR(self.optimizer, self.lambda_lr)
        self.checkpoint_path = config['TRAINING']['CHECKPOINT_PATH']
        

    def train(self):
        raise NotImplementedError 

    def evaluation(self):
        raise NotImplementedError
        
    def save_checkpoint(self, dict_for_updating):
        if not os.isdir(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)
        dict_for_saving = {
            'epoch': self.epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optim.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        for key, value in dict_for_updating.items():
            dict_for_saving[key] = value
        torch.save(dict_for_saving, os.path.join(self.checkpoint_path, "last_model.pth"))
        
    def lambda_lr(self, step):
        warm_up = self.warmup
        step += 1
        return (self.model.d_model ** -.5) * min(step ** -.5, step * warm_up ** -1.5)

    def start(self):
        best_val_score = .0
        patience = 0
        
        for it in range(self.epoch):
            self.train()

            # val scores
            scores = self.evaluate_metrics(self.dev_dataloader)
            val_score = scores[self.score]

            best = False
            if val_score > best_val_score:
                best_val_score = val_score
                patience = 0
                best = True
            else:
                patience += 1

            exit_train = False

            if patience == self.patience:
                exit_train = True

            self.save_checkpoint({
                'best_val_score': best_val_score,
                'patience': patience
            })

            if best:
                copyfile(os.path.join(self.checkpoint_path, "last_model.pth"), 
                         os.path.join(self.checkpoint_path, "best_model.pth"))

            if exit_train:
                break

            self.running_epoch += 1
