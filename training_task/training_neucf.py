import torch
from torch import nn
from tqdm import tqdm
from base_task import BaseTask
from recsys_metrics import rank_report

class TrainingNeuCF(BaseTask):
    def __init__(self, config, model, train_dataloader, dev_dataloader):
        super().__init__(config, model, train_dataloader, dev_dataloader)
        self.loss_fn = nn.L1Loss()
        
    def train(self):
        self.model.to(self.device)
        self.model.train()

        running_loss = 0
        with tqdm(desc='Epoch %d - Training with cross-entropy loss' % self.running_epoch, unit='it', total=len(self.train_dataloader)) as pbar:
            for it, items in enumerate(self.train_dataloader):
                items = items.to(self.device)
                out = self.model(items)  # interacted_rate, trigram_ids
                self.optimizer.zero_grad()
                loss = self.loss_fn(out, items['nli_scores'])
                loss.backward()

                self.optimizer.step()
                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()
                self.scheduler.step()
    
    def evaluation(self):
        gts = []
        gens = []
        self.model.eval()
        with tqdm(desc='Epoch %d - Evaluation' % self.epoch, unit='it', total=len(self.dev_dataloader)) as pbar:
            for it, items in enumerate(self.test_dataloader):
                items = items.to(self.device)
                with torch.inference_mode():
                    outs = self.model(items)
                gts.append(items['labels'])
                gens.append(outs.flatten())
        gts = torch.stack(gts)
        gens = torch.stack(gens)
        scores = rank_report(preds=gens, 
                             target=gts, 
                             k=10, 
                             to_item=True, 
                             name_abbreviation=True,
                             rounding=4)
        
        return scores