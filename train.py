import pandas as pd
import yaml
from data.news_dataset import NewsDataset
from data.test_sample import TestSamples
from model.neucf import NeuCF
from torch.utils.data import DataLoader
from training_task.training_neucf import TrainingNeuCF
import json

with open('/content/DS300/config/config.yaml', 'rb') as f:
    config = yaml.safe_load(f)

df = pd.read_csv('/content/all_data.csv')

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
val = pd.read_csv('data/val.csv')

train_data = NewsDataset(config, train)
val_data = TestSamples(config, val, df)
test_data = TestSamples(config, test, df)

train_loader = DataLoader(train_data,
                          batch_size=config['DATA']['BATCH_SIZE'],
                          collate_fn=train_data.collate_fn)
val_loader = DataLoader(val,
                        batch_size=1,
                        collate_fn=test_data.collate_fn)
test_loader = DataLoader(test_data,
                         batch_size=1,
                         collate_fn=test_data.collate_fn)

model = NeuCF(config)
task = TrainingNeuCF(config, model)
task.start(train_loader, val_loader)

output = task.evaluation(test_loader)
with open('neucf_evaluation.json', 'w') as f:
    json.dump(output, f)