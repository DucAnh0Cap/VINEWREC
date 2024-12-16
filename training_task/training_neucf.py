import torch
from torch import nn
from tqdm import tqdm
from .base_task import BaseTask
from recsys_metrics import rank_report


class TrainingNeuCF(BaseTask):
    def __init__(self, config, model):
        super().__init__(config, model)
        self.loss_fn = nn.L1Loss()
        
    def train(self, train_dataloader):
        self.model.to(self.device)
        self.model.train()

        running_loss = 0
        with tqdm(desc='Epoch %d - Training with L1 loss' % self.running_epoch, unit='it', total=len(train_dataloader)) as pbar:
            for it, items in enumerate(train_dataloader):
                for key, value in items.items():
                    if isinstance(value, torch.Tensor):
                        items[key] = value.to(self.device)
                out = self.model(items)  # interacted_rate, trigram_ids
                self.optimizer.zero_grad()
                loss = self.loss_fn(out.flatten(), items['nli_scores'])
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

    def evaluation(self, val_dataloader):
        gts = []
        gens = []
        self.model.eval()
        with tqdm(desc='Epoch %d - Evaluation' % self.epoch, unit='it', total=len(val_dataloader)) as pbar:
            for it, items in enumerate(val_dataloader):
                for key, value in items.items():
                    if isinstance(value, torch.Tensor):
                        items[key] = value.squeeze().to(self.device)
                with torch.inference_mode():
                    outs = self.model(items).flatten()
                    outs, indices = torch.sort(outs, descending=True)
                    gt = items['labels'][indices]
                gts.append(gt)
                gens.append(outs)

                pbar.update()
        gts = torch.stack(gts)
        gens = torch.stack(gens)
        scores = rank_report(preds=gens,
                             target=gts,
                             k=10,
                             to_item=True,
                             name_abbreviation=True,
                             rounding=10)

        return scores
