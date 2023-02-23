import numpy as np
import torch 
import json
import pickle
import unicodedata
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt
import transformers
from transformers.optimization import get_linear_schedule_with_warmup
from transformers import BertModel, BertTokenizer, DebertaTokenizer, DebertaModel, RobertaTokenizer, RobertaModel, ElectraTokenizer, ElectraModel
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, recall_score, roc_auc_score, precision_score
# from torch_geometric.nn import GCNConv, GATConv, TransformerConv
import pandas as pd
import os
from collections import defaultdict, namedtuple, OrderedDict
from torch.utils.data import Dataset, DataLoader
from itertools import count 
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, Adam, RMSprop
from sklearn.model_selection import train_test_split
from copy import deepcopy
from typing import Union, Callable
from torch import Tensor
from sklearn.utils import shuffle
from typing import Union, Callable, Optional, Tuple
import random
import gdown
import pickle as pk
from sklearn.utils.class_weight import compute_class_weight
# from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('punkt')

seeds = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
seed_idx = 0
seed = seeds[seed_idx]
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

train_data = pd.read_csv('train_all_tasks.csv')
eval_label = pd.read_csv('dev_A.csv')
eval_text = pd.read_csv('dev_task_a_entries.csv')
eval_data = eval_text.merge(eval_label, on='rewire_id')
test_data = pd.read_csv('test_task_a_entries.csv')
gab_unlabeled = pd.read_csv('gab_1M_unlabelled.csv')
reddit_unlabeled = pd.read_csv('reddit_1M_unlabelled.csv')
all_data_edos = pd.read_csv('edos_labelled.csv')
test_data = deepcopy(all_data_edos[all_data_edos['split'] == 'test']).reset_index(drop=True)

from transformers import AutoTokenizer, AutoModel
deberta = 'microsoft/deberta-v3-large' 
roberta = 'roberta-large'
bert = 'bert-large-uncased'
electra = 'google/electra-large-discriminator'
model_name = roberta
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

def tolist(tensor):
  return tensor.detach().cpu().tolist()

def map_names_2_ids(names):
  A = dict()
  for id, name in enumerate(names):
    A[name] = id
  return A

def dist(x1, x2):
  return (x1 - x2).pow(2).sum(-1).sqrt()

def entropy(logits):
  probs = F.softmax(logits, dim=-1)
  ent = -torch.sum((probs * torch.log2(probs)),dim=1)
  return ent

def map_num_2_label(array, dict_map):
  new_dict_map = dict()
  new_array = list()
  for k, v in dict_map.items():
    new_dict_map[v] = k
  for element in array:
    new_array.append(new_dict_map[element])
  return new_array

label_sexist = np.unique(train_data['label_sexist']).tolist()
label_category = np.unique(train_data['label_category']).tolist()
label_vector = np.unique(train_data['label_vector']).tolist()
label_category = [label_category[-1]] + label_category[:-1]
label_vector = [label_vector[-1]] + label_vector[:-1]
label_sexist_map = map_names_2_ids(label_sexist)
label_category_map = map_names_2_ids(label_category)
label_vector_map = map_names_2_ids(label_vector)

train_data['Tag_A'] = [label_sexist_map[i[1]] for i in train_data['label_sexist'].iteritems()]
eval_data['Tag_A'] = [label_sexist_map[i[1]] for i in eval_data['label'].iteritems()]
train_data['Tag_B'] = [label_category_map[i[1]] for i in train_data['label_category'].iteritems()]
train_data['Tag_C'] = [label_vector_map[i[1]] for i in train_data['label_vector'].iteritems()]
test_data['Tag_A'] = [label_sexist_map[i[1]] for i in test_data['label_sexist'].iteritems()]
labels_names = label_sexist

train_dataframe = train_data
eval_dataframe = eval_data
test_dataframe = test_data
class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=train_dataframe['Tag_A'].values).tolist()

class SexistDataset(Dataset):

  def __init__(self, dataframe, tokenizer, max_length=100, use_label=True):
    self.dataframe = dataframe
    self.tokenizer = tokenizer
    self.max_length = max_length
    self.labels_names = f'{tokenizer.sep_token}'.join(labels_names)
    self.use_label = use_label

    self.labels_tokens = []
    for label_name in labels_names:
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

    if self.use_label:
      labels_A = torch.LongTensor([sample['Tag_A']])
      tokenized_text['Tag_A'] = labels_A

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


def exists(value):
    return value is not None


def default(value, default):
    if exists(value):
        return value
    return default


def inf_norm(x):
    return torch.norm(x, p=float("inf"), dim=-1, keepdim=True)


def kl_loss(input, target, reduction="batchmean"):
    return F.kl_div(
        F.log_softmax(input, dim=-1),
        F.softmax(target, dim=-1),
        reduction=reduction,
    )


def sym_kl_loss(input, target, reduction="batchmean", alpha=1.0):
    return alpha * F.kl_div(
        F.log_softmax(input, dim=-1),
        F.softmax(target.detach(), dim=-1),
        reduction=reduction,
    ) + F.kl_div(
        F.log_softmax(target, dim=-1),
        F.softmax(input.detach(), dim=-1),
        reduction=reduction,
    )


def js_loss(input, target, reduction="batchmean", alpha=1.0):
    mean_proba = 0.5 * (
        F.softmax(input.detach(), dim=-1) + F.softmax(target.detach(), dim=-1)
    )
    return alpha * (
        F.kl_div(F.log_softmax(input, dim=-1), mean_proba, reduction=reduction)
        + F.kl_div(F.log_softmax(target, dim=-1), mean_proba, reduction=reduction)
    )


class SMARTLoss(nn.Module):
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable,
        loss_last_fn: Callable = None, 
        norm_fn: Callable = inf_norm, 
        num_steps: int = 1,
        step_size: float = 1e-3, 
        epsilon: float = 1e-6,
        noise_var: float = 1e-5
    ) -> None:
        super().__init__()
        self.model = model 
        self.loss_fn = loss_fn
        self.loss_last_fn = default(loss_last_fn, loss_fn)
        self.norm_fn = norm_fn
        self.num_steps = num_steps 
        self.step_size = step_size
        self.epsilon = epsilon 
        self.noise_var = noise_var
     
    @torch.enable_grad()   
    def forward(self, embed, state):
        noise = torch.randn_like(embed, requires_grad = True) * self.noise_var 
        
        # Indefinite loop with counter 
        for i in count():
            # Compute perturbed embed and states 
            embed_perturbed = embed + noise 
            state_perturbed = self.model(embed_perturbed) 
            # Return final loss if last step (undetached state)
            if i == self.num_steps: 
                return self.loss_last_fn(state_perturbed, state) 
            # Compute perturbation loss (detached state)
            loss = self.loss_fn(state_perturbed, state.detach())
            # Compute noise gradient ∂loss/∂noise
            noise_gradient, = torch.autograd.grad(loss, noise)
            # Move noise towards gradient to change state as much as possible 
            step = noise + self.step_size * noise_gradient 
            # Normalize new noise step into norm induced ball 
            step_norm = self.norm_fn(step)
            noise = step / (step_norm + self.epsilon)
            # Reset noise gradients for next step
            noise = noise.detach().requires_grad_()




class ExtractedRoBERTa(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.roberta = model
        self.layers = model.encoder.layer  
        self.attention_mask = None 
        self.num_layers = len(self.layers) - 1 
        self.head = nn.Linear(self.roberta.config.hidden_size, 2)

    def forward(self, hidden, with_hidden_states = False, start_layer = 0, return_all=False):
        """ Forwards the hidden value from self.start_layer layer to the logits. """
        hidden_states = [] 
        
        for layer_id, layer in enumerate(self.layers[start_layer:]):
            hidden = layer(hidden, attention_mask = self.attention_mask)[0]

            if layer_id in list(range(20, 24)):
              hidden_states += [hidden]

        logits = self.head(hidden[:, 0, :]) 

        if return_all:
          return logits, hidden_states
        else:
          return logits


    def get_embeddings(self, input_ids):
        """ Computes first embedding layer given inputs_ids """ 
        return self.roberta.embeddings(input_ids)

    def set_attention_mask(self, attention_mask):
        """ Sets the correct mask on all subsequent forward passes """ 
        self.attention_mask = self.roberta.get_extended_attention_mask(
            attention_mask, 
            input_shape = attention_mask.shape, 
            # device = attention_mask.device
        ) # (b, 1, 1, s) 


def ntxent(logits, labels, temp=.07):
  def ntx_loss(a, p, n, temp=temp):
    a = a.unsqueeze(0) if a.dim() == 1 else a
    p = p.unsqueeze(0) if p.dim() == 1 else p
    n = n.unsqueeze(0) if n.dim() == 1 else n
    assert a.dim() == 2
    assert p.dim() == 2
    assert n.dim() == 2
    a_p = a
    a_n = a.repeat(n.shape[0], 1)
    p_sim = F.cosine_similarity(a_p, p, dim=-1) / temp
    n_sim = F.cosine_similarity(a_n, n, dim=-1) / temp

    # apply numeric stability
    max_val = torch.max(n_sim).detach()
    numerator = torch.exp(p_sim - max_val)
    denominator = torch.exp(n_sim - max_val).sum()
    loss = -torch.log(numerator / (denominator + numerator) + 1e-6)
    if loss.isnan():
      print(numerator, denominator)
      print(p_sim)
      print(len(n))
    # print(loss)
    return loss.mean()

  def dist(x1, x2):
    return (x1 - x2).pow(2).sum(-1).sqrt()

  con_losses = list()
  for i, (logit, label) in enumerate(zip(logits, labels)):
    ps = (labels == label)
    ns = (labels != label).nonzero().view(-1)
    ps[i] = False
    ps = ps.nonzero().view(-1)
    if len(ns):
      for p in ps:
        a_logit = logits[i]
        p_logit = logits[p]
        ns_logit = logits[ns]
        A = ntx_loss(a_logit, p_logit, ns_logit)
        con_losses.append(A)

  if len(con_losses) > 0:
    all_con_loss = torch.stack(con_losses).mean()
  else:
    all_con_loss = torch.tensor(0.)
  return all_con_loss


def train(dataloader, model, device, loss_fn, optimizer, scheduler, stage, ul_dataset, use_adv=True, use_vadv=True, 
          use_ul=False, use_contrastive=True, con_weight=.5, vat_weight=.5, ul_weight=.5, adv_use_every_layer=True):
  
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
    labels = data.pop('Tag_A').to(device).view(-1)
    for key in data:
      data[key] = data[key].to(device).view(c_batch_size, -1)

    embeddings = model.get_embeddings(data['input_ids'].to(device))
    model.set_attention_mask(data['attention_mask'].to(device))
    logits = model(embeddings)

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

            embeddings = model.get_embeddings(data['input_ids'].to(device))
            model.set_attention_mask(data['attention_mask'].to(device))
            logits = model(embeddings)
            loss_adv = loss_fn(logits, labels)
            loss_adv.backward()
            loss_collection[1].append(loss_adv.item())
        pgd.restore() 

    if use_contrastive:
      embeddings = model.get_embeddings(data['input_ids'].to(device))
      model.set_attention_mask(data['attention_mask'].to(device))
      logits, hidden_states = model(embeddings, return_all=True)
      con_losses = []
      for hidden_idx, hidden_state in enumerate(hidden_states[::-1]):
        con_losses.append(ntxent(logits, labels) * con_weight * (1/(hidden_idx + 1)))
      con_loss = torch.stack(con_losses).mean()
      if con_loss.requires_grad:
        con_loss.backward()
      loss_collection[2].append(con_loss.item())

    if use_vadv:
      vat_loss_fn = SMARTLoss(model = model, loss_fn = kl_loss, loss_last_fn = sym_kl_loss)
      # Compute VAT loss
      embeddings = model.get_embeddings(data['input_ids'].to(device))
      model.set_attention_mask(data['attention_mask'].to(device))
      logits = model(embeddings)
      vat_loss = vat_loss_fn(embeddings, logits) 
      # Merge losses 
      vat_loss = vat_weight * vat_loss
      vat_loss.backward()
      loss_collection[3].append(vat_loss.item())

    if use_ul:
      ul_data = ul_dataset.next()
      c_batch_size = ul_data['input_ids'].shape[0]
      for key in ul_data:
        ul_data[key] = ul_data[key].to(device).view(c_batch_size, -1)

      ul_embeddings = model.get_embeddings(ul_data['input_ids'].to(device))
      model.set_attention_mask(ul_data['attention_mask'].to(device))
      ul_logits = model(ul_embeddings)

      vat_loss_fn = SMARTLoss(model = model, loss_fn = kl_loss, loss_last_fn = sym_kl_loss)
      vat_loss = vat_loss_fn(ul_embeddings, ul_logits) 
      ul_loss = ul_weight * vat_loss
      loss_collection[4].append(ul_loss.item())
      ul_loss.backward()


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
      Tag_A = data.pop('Tag_A').to(device).view(-1)
      embeddings = model.get_embeddings(data['input_ids'].to(device))
      model.set_attention_mask(data['attention_mask'].to(device))
      logits = model(embeddings)
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
      embeddings = model.get_embeddings(data['input_ids'].to(device))
      model.set_attention_mask(data['attention_mask'].to(device))
      logits = model(embeddings)
      preds = tolist(logits.argmax(1).view(-1))
      all_preds.extend(preds)
  return all_preds


checkpoint_dir = 'Checkpoint_A_contrastive'

if not os.path.exists(checkpoint_dir):
  os.mkdir(checkpoint_dir)
filename = os.path.join(checkpoint_dir, 'best_ch.pt')

def save_model(epoch, model, optimizer, scheduler, f1_list):
  torch.save(
      {'epoch': epoch,
       'model_state_dict': model.state_dict(),
       'optimizer_state_dict': optimizer.state_dict(),
       'scheduler_state_dict': scheduler.state_dict(),
       'f1_list': f1_list
       },
        filename)

def load_model():
  if os.path.exists(filename):
    saved_dict = torch.load(filename)
    return True, saved_dict
  else:
    return False, None


epochs = 10
lr = 4e-6
beta_1 = .9
beta_2 = .999
eps = 1e-6
log_step = 100
batch_size = 10
weight_decay = 1e-2
max_length = 70

vat_weight = .5
ul_weight = .5
ent_weight = .5
label_smoothing = .0


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# sexist_model = SexistModel(deepcopy(model)).to(device)
sexist_model = ExtractedRoBERTa(deepcopy(model)).to(device)
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(device), label_smoothing=label_smoothing).to(device)
loss_collection = []

train_dataset = SexistDataset(train_dataframe, tokenizer, max_length)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

eval_dataset = SexistDataset(eval_dataframe, tokenizer, max_length)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

test_dataset = SexistDataset(test_dataframe, tokenizer, max_length)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

optimization_steps = epochs * len(train_dataloader)
warmup_ratio = .0
warmup_steps = int(optimization_steps * warmup_ratio)


optimizer = AdamW(sexist_model.parameters(), lr=lr, betas=(beta_1,beta_2), eps=eps, weight_decay=weight_decay)
scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=warmup_steps, 
    num_training_steps=optimization_steps)

best_f1 = 0.
all_f1 = list()
best_model = None
best_preds = None
transformers.logging.set_verbosity_error()
start_epoch = 0

checkpoint_avl, saved_dict = load_model()
print(f'Checkpoit is {checkpoint_avl}')
if checkpoint_avl:
  start_epoch = saved_dict['epoch']
  model_state_dict = saved_dict['model_state_dict']
  optimizer_state_dict = saved_dict['optimizer_state_dict']
  scheduler_state_dict = saved_dict['scheduler_state_dict']
  all_f1 = saved_dict['f1_list']
  best_f1 = max(all_f1)

  sexist_model.load_state_dict(model_state_dict)
  optimizer.load_state_dict(optimizer_state_dict)
  scheduler.load_state_dict(scheduler_state_dict)

print('model loaded')


for epoch in range(start_epoch, epochs):
  train(train_dataloader, sexist_model, device, loss_fn, optimizer, scheduler, 1, un)
  preds_A_eval = eval(eval_dataloader, sexist_model, device)
  f1_macro_A_eval = f1_score(eval_dataframe['Tag_A'].values.tolist(), preds_A_eval, average='macro')
  all_f1.append(f1_macro_A_eval)
  if f1_macro_A_eval > best_f1:
    best_f1 = f1_macro_A_eval
    best_preds = preds_A_eval
    save_model(epoch + 1, sexist_model, optimizer, scheduler, all_f1)

  tw = f'EPOCH [{epoch + 1}/{epochs}] | Current F1-Macro {round(f1_macro_A_eval * 100, 2)}\n'
  with open(f'pred.txt', 'a') as f:
    f.write(tw)
  print(f'EPOCH [{epoch + 1}/{epochs}] | Current F1-Macro {round(f1_macro_A_eval * 100, 2)}')
  print(f'EPOCH [{epoch + 1}/{epochs}] | Best F1-Macro {round(best_f1 * 100, 2)}')

with open(f'pred_file_{seed}.pk', 'wb') as f:
  pickle.dump(best_preds, f)

_, saved_dict = load_model()
sexist_model.load_state_dict(saved_dict['model_state_dict'])

preds_A_test = eval(test_dataloader, sexist_model, device)
f1_macro_A_test = f1_score(test_dataframe['Tag_A'].values.tolist(), preds_A_test, average='macro')

tw = f'Test F1-Macro {round(f1_macro_A_test * 100, 2)}\n'
with open(f'test_result.txt', 'a') as f:
  f.write(tw)
