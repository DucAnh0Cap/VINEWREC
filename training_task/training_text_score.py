import torch
from tqdm import tqdm
from utils import compute_multiclass_metrics
from .base_task import BaseTask


class TrainingTextScore(BaseTask):
    def __init__(self, config, model):
        super().__init__(config, model)

    def train(self, train_dataloader):
        self.model.to(self.device)
        self.model.train()

        running_loss = 0
        with tqdm(desc='Epoch %d - Training with cross-entropy loss' % self.running_epoch, unit='it', total=len(train_dataloader)) as pbar:
            for it, items in enumerate(train_dataloader):
                for key, value in items.items():
                    if isinstance(value, torch.Tensor):
                        items[key] = value.to(self.device)
                out = self.model(items).flatten()  # interacted_rate, trigram_ids
                self.optimizer.zero_grad()
                loss = self.loss_fn(out, items['interacted_categories'])
                loss.backward()

                self.optimizer.step()
                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()
                self.scheduler.step()
                
    def lambda_lr(self, step):
        warm_up = self.warmup
        step += 1
        return (self.model.latent_dim_mlp ** -.5) * min(step ** -.5, step * warm_up ** -1.5)
    
    def evaluation(self, dev_dataloader):
        gts = []
        gens = []
        self.model.eval()
        with tqdm(desc='Epoch %d - Evaluation' % self.running_epoch, unit='it', total=len(dev_dataloader)) as pbar:
            for it, items in enumerate(dev_dataloader):
                for key, value in items.items():
                    if isinstance(value, torch.Tensor):
                        items[key] = value.to(self.device)
                with torch.inference_mode():
                    outs = self.model(items)
                gts.extend(items['interacted_categories'].cpu())
                gens.extend(outs.cpu())
        scores = compute_multiclass_metrics(gens, gts)
        pbar.update()
        
        return scores
