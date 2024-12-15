import torch
from tqdm import tqdm
from utils import compute_multiclass_metrics
from base_task import BaseTask


class TrainingTextScore(BaseTask):
    def __init__(self, config, model, train_dataloader, dev_dataloader):
        super().__init__(config, model, train_dataloader, dev_dataloader)

    def train(self):
        self.model.to(self.device)
        self.model.train()

        running_loss = 0
        with tqdm(desc='Epoch %d - Training with cross-entropy loss' % self.running_epoch, unit='it', total=len(self.train_dataloader)) as pbar:
            for it, items in enumerate(self.train_dataloader):
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

    def evaluation(self):
        gts = []
        gens = []
        self.model.eval()
        with tqdm(desc='Epoch %d - Evaluation' % self.epoch, unit='it', total=len(self.dev_dataloader)) as pbar:
            for it, items in enumerate(self.test_dataloader):
                for key, value in items.items():
                    if isinstance(value, torch.Tensor):
                        items[key] = value.to(self.device)
                with torch.inference_mode():
                    outs = self.model(items)
                gts.extend(items['interacted_categories'].cpu())
                gens.extend(outs.cpu())
        scores = compute_multiclass_metrics(gens, gts)
        
        return scores
