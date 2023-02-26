import numpy as np
import torch 
import json
import pickle
import unicodedata
import re
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import transformers
from transformers.optimization import get_linear_schedule_with_warmup
from transformers import BertModel, BertTokenizer, DebertaTokenizer, DebertaModel, RobertaTokenizer, RobertaModel, ElectraTokenizer, ElectraModel
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, recall_score, roc_auc_score, precision_score
import pandas as pd
import os
from collections import defaultdict, namedtuple, OrderedDict
from torch.utils.data import Dataset, DataLoader
from itertools import count 
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, Adam, RMSprop
from copy import deepcopy
from sklearn.utils import shuffle
from typing import Union, Callable
import random
import gdown
from torch import Tensor
from typing import Optional, Tuple
import pickle as pk
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
# from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('punkt')

seeds = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
seed_idx = 1
seed = seeds[seed_idx]
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

from transformers import AutoTokenizer, AutoModel
deberta = 'microsoft/deberta-v3-large' 
roberta = 'roberta-large'
model_name = deberta
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_hidden_states=True)


train_data = pd.read_csv('train_all_tasks.csv')
eval_data_A = pd.read_csv('dev_task_a_entries.csv')
eval_data_B = pd.read_csv('dev_task_b_entries.csv')
eval_data_C = pd.read_csv('dev_task_c_entries.csv')

test_B = pd.read_csv('test_task_b_entries.csv')
test_C = pd.read_csv('test_task_c_entries.csv')


def tolist(tensor):
  return tensor.detach().cpu().tolist()

def map_names_2_ids(names):
  A = dict()
  B = dict()
  for id, name in enumerate(names):
    A[name] = id
    B[id] = name
  return A, B

def dist(x1, x2):
  return (x1 - x2).pow(2).sum(-1).sqrt()

def entropy(logits):
  probs = F.softmax(logits, dim=-1)
  ent = -torch.sum((probs * torch.log2(probs)),dim=1)
  return ent

train_data = train_data[train_data['label_sexist'] == 'sexist'].reset_index(drop=True)

label_category_raw = np.unique(train_data['label_category']).tolist()
label_category_map, category_label_map = map_names_2_ids(label_category_raw)
train_data['Tag_B'] = [label_category_map[i[1]] for i in train_data['label_category'].iteritems()]
label_category = list(label_category_map.keys())
label_category = list(map(lambda x: re.sub('^\d+\.\d*', '', x).strip(), label_category))
eval_label_B = pd.read_csv('dev_task_b_labels.csv')
eval_B = eval_data_B.merge(eval_label_B, on='rewire_id')
eval_B['Tag_B'] = [label_category_map[i[1]] for i in eval_B['label'].iteritems()]
num_labels_B = len(train_data['label_category'].unique())

label_vector_raw = np.unique(train_data['label_vector']).tolist()
label_vector_map, vector_label_map = map_names_2_ids(label_vector_raw)
train_data['Tag_C'] = [label_vector_map[i[1]] for i in train_data['label_vector'].iteritems()]
label_vector = list(label_vector_map.keys())
label_vector = list(map(lambda x: re.sub('^\d+\.\d*', '', x).strip(), label_vector))
eval_label_C = pd.read_csv('dev_task_c_labels.csv')
eval_C = eval_data_C.merge(eval_label_C, on='rewire_id')
eval_C['Tag_C'] = [label_vector_map[i[1]] for i in eval_C['label'].iteritems()] 
num_labels_C = len(train_data['label_vector'].unique())

train_dataframe = train_data
eval_dataframe = eval_B
test_dataframe = test_B
new_l = label_category

class_weights = compute_class_weight(class_weight='balanced', classes=np.array(list(range(num_labels_B))), y=train_data['Tag_B'].values.tolist()).tolist()

class SexistDataset(Dataset):

  def __init__(self, dataframe, tokenizer, max_length=100, is_test=False):
    self.dataframe = dataframe
    self.tokenizer = tokenizer
    self.max_length = max_length
    self.labels_names = f'{tokenizer.sep_token}'.join(new_l)
    self.is_test = is_test

    self.labels_tokens = []
    for label_name in new_l:
      label_tokens = tokenizer(label_name, add_special_tokens=False)
      self.labels_tokens.append(label_tokens['input_ids'])

  def __len__(self):
    return len(self.dataframe)

  def __getitem__(self, idx):
    sample = self.dataframe.loc[idx]
    tokenized_text = tokenizer(
          sample['text'],
          max_length=self.max_length,
          padding='max_length',
          truncation='only_first',
          return_tensors='pt')

    # find the first token of labels
    input_ids = tokenized_text['input_ids']
    labels_start = (input_ids == tokenizer.sep_token_id).nonzero().contiguous().view(-1).tolist()[1] + 2

    labels_tokens_span = []
    c_token = labels_start
    # print(labels_start)
    for label_tokens in self.labels_tokens:

      labels_tokens_span.append([c_token, c_token + len(label_tokens) - 1])
      c_token += len(label_tokens) + 1
    tokenized_text['labels_tokens_span'] = torch.tensor(labels_tokens_span)
    if not self.is_test:
      labels_B = torch.LongTensor([sample['Tag_B']])
      tokenized_text['Tag_B'] = labels_B
    return tokenized_text



class PGD():

    def __init__(self, model,emb_name,epsilon=1.,alpha=0.3):
        # The emb_name parameter should be replaced with the parameter name of the embedding in your model
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}

    # adversarial training : attack to change embedding abit with regards projected gradiant descent
    def attack(self,first_strike=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if first_strike:
                    # print('tt', param.data)
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    # Compute new params
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, self.epsilon)

    # Restore to the back-up embeddings
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    # Project Gradiant Descent
    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    # Back-up parameters
    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'pooler' not in name:
                self.grad_backup[name] = param.grad.clone()

    # Restore grad parameters
    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'pooler' not in name:
                param.grad = self.grad_backup[name]





class SexistModel(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.transformer = model
        hidden_size = self.transformer.config.hidden_size
        self.dropout = nn.Dropout(p=.3)
        self.head = nn.Linear(hidden_size, num_labels_B)

    def integrate(self, batch_output, batch_labels_tokens_span):
      batch_size = batch_output.shape[0]
      integrated_batch = []
      for i in range(batch_size):
        integrated_sample_labels = []
        output = batch_output[i]
        labels_tokens_span = batch_labels_tokens_span[i]
        for label_tokens_span in labels_tokens_span:
          integrated_label = output[label_tokens_span[0].item(): label_tokens_span[1].item() + 1].mean(0).view(-1)
          assert integrated_label.shape[0] == self.transformer.config.hidden_size
          integrated_sample_labels.append(integrated_label)
        integrated_sample_labels = torch.stack(integrated_sample_labels)
        integrated_batch.append(integrated_sample_labels)
      integrated_batch = torch.stack(integrated_batch)
      return integrated_batch

    def forward(self, x, batch_labels_tokens_span, vat=False, attention_mask=None):
        if vat:
          hidden = self.transformer(inputs_embeds=x, attention_mask=attention_mask).last_hidden_state
        else:
          hidden = self.transformer(**x).last_hidden_state
        cls = hidden[:, 0, :]
        x = self.head(cls)
        x = x.view(-1, num_labels_B)
        return x, hidden




def train(dataloader, model, device, loss_fn, optimizer, scheduler, stage, ul_dataset, use_contrastive=False,
          use_adv=True, use_vadv=False, use_ul=False, vat_weight=.5, ul_weight=.5, con_weight=.5, adv_use_every_layer=True):
  
  model.train()
  named_weights = [n for n, _ in model.named_parameters() if 'dense.weight' in n and 'pooler' not in n] + ["word_embeddings."]
  loss_collection = [[], [], [], [], []]
  for step, data in enumerate(dataloader):

    if adv_use_every_layer:
      rand_layer = random.sample(named_weights, 1)[0] 
      adv_layer = rand_layer
    else:
      adv_layer = "word_embeddings."
    pgd = PGD(
      model=model,
      emb_name=adv_layer
    )


    c_batch_size = data['input_ids'].shape[0]
    labels = data.pop('Tag_B').to(device).view(-1)
    for key in data:
      data[key] = data[key].to(device).view(c_batch_size, -1)
    batch_labels_tokens_span = data.pop('labels_tokens_span').view(-1, num_labels_B, 2)

    logits, _ = model(data, batch_labels_tokens_span)

    ce_loss = loss_fn(logits, labels)
    ce_loss.backward()
    loss_collection[0].append(ce_loss.item())


    
    if use_adv:
      # PGD Start
        pgd.backup_grad()
        attack_times = 2
        for attack_time in range(attack_times):
            # Add adversarial perturbation to the embedding, backup param.data during the first attack
            pgd.attack(first_strike=(attack_time==0))
            if attack_time != attack_times-1:
              model.zero_grad()
            else:
              pgd.restore_grad()
            logits_adv, _ = model(data, batch_labels_tokens_span)
            loss_adv = loss_fn(logits_adv, labels)
            loss_collection[1].append(loss_adv.item())
            loss_adv.backward()
        # Restore embedding parameters
        pgd.restore() 




    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()
    
    if len(loss_collection[0]) % log_step == 0:
      print(f'EPOCH [{epoch + 1}/{epochs}] | STEP [{step + 1}/{len(train_dataloader)}] | CE Loss {round(sum(loss_collection[0]) / (len(loss_collection[0]) + 1e-8), 4)}')
      print(f'EPOCH [{epoch + 1}/{epochs}] | STEP [{step + 1}/{len(train_dataloader)}] | ADV Loss {round(sum(loss_collection[1]) / (len(loss_collection[1]) + 1e-8), 4)}')
      print(f'EPOCH [{epoch + 1}/{epochs}] | STEP [{step + 1}/{len(train_dataloader)}] | CON Loss {round(sum(loss_collection[2]) / (len(loss_collection[2]) + 1e-8), 4)}')
      print(f'EPOCH [{epoch + 1}/{epochs}] | STEP [{step + 1}/{len(train_dataloader)}] | VAT Loss {round(sum(loss_collection[3]) / (len(loss_collection[3]) + 1e-8), 4)}')
      print(f'EPOCH [{epoch + 1}/{epochs}] | STEP [{step + 1}/{len(train_dataloader)}] | UL Loss {round(sum(loss_collection[4]) / (len(loss_collection[4]) + 1e-8), 4)}')
      print('------------------------------------------------')
      loss_collection = [[] for _ in range(5)]


def eval(dataloader, model, device):
  with torch.no_grad():
    model.eval()
    all_preds = list()

    for data in dataloader:
      c_batch_size = data['input_ids'].shape[0]
      for key in data:
        data[key] = data[key].to(device).view(c_batch_size, -1)
      batch_labels_tokens_span = data.pop('labels_tokens_span').view(-1, num_labels_B, 2)
      Tag_B = data.pop('Tag_B').to(device).view(-1)
      logits, _ = model(data, batch_labels_tokens_span)
      preds = tolist(logits.argmax(1).view(-1))
      all_preds.extend(preds)
  return all_preds


def test(dataloader, model, device):
  with torch.no_grad():
    model.eval()
    all_preds = list()

    for data in dataloader:
      c_batch_size = data['input_ids'].shape[0]
      for key in data:
        data[key] = data[key].to(device).view(c_batch_size, -1)
      batch_labels_tokens_span = data.pop('labels_tokens_span').view(-1, num_labels_B, 2)
      logits, _ = model(data, batch_labels_tokens_span)
      preds = tolist(logits.argmax(1).view(-1))
      all_preds.extend(preds)
  return all_preds

epochs = 30
lr = 1e-5
beta_1 = .9
beta_2 = .999
eps = 1e-6
log_step = 100
batch_size = 10
weight_decay = 9e-3
max_length = 70
loss_file = 'loss.txt'
eval_file = 'eval.txt'

vat_weight = .5
ul_weight = .5
ent_weight = .5


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# sexist_model = ExtractedRoBERTa(deepcopy(model)).to(device)
sexist_model = SexistModel(deepcopy(model)).to(device)
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(device)).to(device)
loss_collection = []

train_dataset = SexistDataset(train_dataframe, tokenizer, max_length)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

eval_dataset = SexistDataset(eval_dataframe, tokenizer, max_length)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

test_dataset = SexistDataset(test_dataframe, tokenizer, max_length, is_test=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

opt_step = 0
optimization_steps = epochs * len(train_dataloader)
warmup_ratio = .0
warmup_steps = int(optimization_steps * warmup_ratio)


optimizer = AdamW(sexist_model.parameters(), lr=lr, betas=(beta_1,beta_2), eps=eps, weight_decay=weight_decay)
scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=warmup_steps, 
    num_training_steps=optimization_steps)

best_f1 = 0.
best_model = None
transformers.logging.set_verbosity_error()



checkpoint_dir = 'Task_B/'
filename = os.path.join(checkpoint_dir, 'best_ch.pt')


try:
    os.rmdir(checkpoint_dir)
except:
    pass
if not os.path.exists(checkpoint_dir):
  os.mkdir(checkpoint_dir)

def save_model(epoch, model, optimizer, scheduler):
  filename = os.path.join(checkpoint_dir, 'best_ch.pt')
  torch.save(
      {'epoch': epoch,
       'model_state_dict': model.state_dict(),
       'optimizer_state_dict': optimizer.state_dict(),
       'scheduler_state_dict': scheduler.state_dict()}, 
        filename)

def load_model():
  if os.path.exists(filename):
    saved_dict = torch.load(filename)
    return True, saved_dict
  else:
    return False, None
    

def early_stop(scores, current_score, patience, best_f1):
  if len(scores) < patience:
    return False
  else:
    for score in scores[-patience: ]:
      if score >= best_f1:
        return False
    return True

all_f1 = list()
patience = 4

for epoch in range(epochs):
  train(train_dataloader, sexist_model, device, loss_fn, optimizer, scheduler, 1, None)
  preds_B_eval = eval(eval_dataloader, sexist_model, device)
  f1_macro_B_eval = f1_score(eval_dataframe['Tag_B'].values.tolist(), preds_B_eval, average='macro')
  all_f1.append(f1_macro_B_eval)
  if f1_macro_B_eval > best_f1:
    best_f1 = f1_macro_B_eval
    best_preds = preds_B_eval
    save_model(epoch + 1, sexist_model, optimizer, scheduler)

  print(f'EPOCH [{epoch + 1}/{epochs}] | Current F1-Macro {round(f1_macro_B_eval * 100, 2)}')
  print(f'EPOCH [{epoch + 1}/{epochs}] | Best F1-Macro {round(best_f1 * 100, 2)}')
  print(confusion_matrix(eval_dataframe['Tag_B'].values.tolist(), preds_B_eval))

  if early_stop(all_f1, f1_macro_B_eval, patience, best_f1):
    break
  else:
    print('not early stopping')


_, saved_dict = load_model()
sexist_model.load_state_dict(saved_dict['model_state_dict'])
preds_B_test = test(test_dataloader, sexist_model, device)
test_preds_names = [category_label_map[label_id] for label_id in preds_B_test]
test_B['label_pred'] = test_preds_names
test_B.to_csv('test_B_preds.csv')
