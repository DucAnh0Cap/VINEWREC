import torch
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm
from torchmetrics import Accuracy, Recall, Precision
from torchmetrics.classification import MulticlassF1Score

class TrainingTextScore():
    def __init__(self, config, model, train_dataloader, test_dataloader):
        super().__init__()
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(params=self.model.parameters(),
                                    lr=config.TRAINING.learning_rate)

        self.epoch = config.TRAINING.epoch
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.device = config.TRAINING.device
        self.num_classes = config.DATA.num_classes
        
        self.accuracy_fn = Accuracy(task="multiclass",
                                    num_clases=self.num_classes)
        self.recall_fn = Recall(task="multiclass",
                                average='micro',
                                num_classes=self.num_classes)
        self.precision_fn = Precision(task="multiclass",
                                      average='micro',
                                      num_classes=self.num_classes)
        self.f1_fn = MulticlassF1Score(num_classes=self.num_classes,
                                       average=None)

    def train(self):
        self.model.to(self.device)
        self.model.train()

        running_loss = 0
        with tqdm(desc='Epoch %d - Training with cross-entropy loss' % self.epoch, unit='it', total=len(self.train_dataloader)) as pbar:
            for it, items in enumerate(self.train_dataloader):
                items = items.to(self.device)
                out = self.model(items)  # interacted_rate, trigram_ids
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
        with tqdm(desc='Epoch %d - Evaluation' % self.epoch, unit='it', total=len(self.test_dataloader)) as pbar:
            for it, items in enumerate(self.test_dataloader):
                items = items.to(self.device)
                with torch.inference_mode():
                    outs = self.model(items)
                gts.extend(list(items['interacted_categories'].numpy()))
                gens.extend(list(outs.numpy()))
        accuracy = self.accuracy_fn(gens, gts)
        recall = self.recall_fn(gens, gts)
        precision = self.precision_fn(gens, gts)
        f1 = self.f1_fn(gens, gts)
        
        return {
            'accuracy': accuracy,
            'recall': recall,
            'precision': precision,
            'f1': f1,
        }