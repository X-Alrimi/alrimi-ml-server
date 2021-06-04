import torch 
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from kobert_transformers import get_tokenizer, get_kobert_model 

import random
import numpy as np
import pandas as pd

from news_dataset import NewsDataset
from model import BERTNewsClassifier
from loss import LabelSmoothingLoss

from utils import get_f1_score


random.seed(1)
np.random.seed(1)
np.random.RandomState(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.enabled=False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic=True

##0. Set Hyperparameter 
train_fn = './data/train_data.csv'
valid_fn = './data/valid_data.csv'

embed_size = 512
batch_size = 12

num_workers = 5

epoch_num = 20

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

learning_rate = 5e-5
warmup_ratio = 0.01
max_grad_norm = 1e+8

num_classes=2

hyper_params = {
    'train_fn' : train_fn, 'valid_fn' : valid_fn, 'embed_size' : embed_size, 'batch_size' : batch_size,
    'num_workers' : num_workers, 'num_classes' : num_classes,
    'epoch_num' : 10, 'device' : device, 'learning_rate' : learning_rate, 
    'warmup_ratio' : 0.01, 'max_grad_norm' : max_grad_norm,
}



## 1. read data 
train_data = pd.read_csv(train_fn)
valid_data = pd.read_csv(valid_fn)


##2. Create Dataset/DataLoader  
tokenizer = get_tokenizer()
news_dataset = NewsDataset(train_data['input'], train_data['label'], tokenizer, embed_size, batch_size)
news_dataloader = DataLoader(news_dataset, batch_size=batch_size, num_workers=num_workers)

valid_dataset = NewsDataset(valid_data['input'], valid_data['label'], tokenizer, embed_size, batch_size)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers)


##3. Create Model 
bertmodel = get_kobert_model()
model = BERTNewsClassifier(bertmodel, num_classes=num_classes).to(device)


##4. Create Optimizer/Scheduler 
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]


crit = LabelSmoothingLoss(num_classes, 0.25)
optimizer = AdamW(optimizer_grouped_parameters, lr = learning_rate)

t_total = len(news_dataloader) * epoch_num
warmup_step = int(t_total * warmup_ratio)

scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

hyper_params['no_decay'] = no_decay 
hyper_params['optimizer_grouped_params'] = optimizer_grouped_parameters
hyper_params['t_total'] = t_total 
hyper_params['warmpup_step'] = warmup_step


## 5. Train and Validate
for cur_epoch in range(epoch_num) : 
  model.train() 
  for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(news_dataloader):
        optimizer.zero_grad()
        if type(token_ids) == list : 
          token_ids = torch.stack(token_ids).transpose(0,1).contiguous()
        if token_ids.dim() == 1 : 
          token_ids = token_ids.unsqueeze(0)
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out,_ = model(token_ids, valid_length, segment_ids)
        loss = crit(out, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        ## Need to calc F1-Score 
        preds = [ int(torch.argmax(cur_out)) for cur_out in out]
        label = [ int(cur) for cur in label]
        label_1_num = sum([ cur==1 for cur in label])

        cur_acc =  sum([c_label==c_pred for c_label, c_pred in zip(label,preds) ]) /len(label)

        print("epoch {} batch id {}/{} loss {} cur_acc {} ".format(cur_epoch+1, batch_id+1, len(news_dataloader), loss.data.cpu().numpy(), cur_acc))

  test_result = []
  
  model.eval() 
  for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(valid_dataloader):
    with torch.no_grad() :
      if type(token_ids) == list : 
        token_ids = torch.stack(token_ids).transpose(0,1).contiguous()
      if token_ids.dim() == 1 : 
        token_ids = token_ids.unsqueeze(0)
      token_ids = token_ids.long().to(device)
      segment_ids = segment_ids.long().to(device)
      valid_length= valid_length
      label = label.long().to(device)
      out,_ = model(token_ids, valid_length, segment_ids)
      preds = [ int(torch.argmax(cur_out)) for cur_out in out]
      label = [ int(cur) for cur in label]
      test_result.extend([ (cur_pred,cur_label) for cur_pred, cur_label in zip(preds, label)])

      cur_acc +=  sum([c_label==c_pred for c_label, c_pred in zip(label,preds) ]) /len(label)
  
  recall, precision, f1_score = get_f1_score(test_result)
  valid_acc = cur_acc/len(valid_dataloader)
  print("epoch {} recall : {}  precision : {} f1_score : {} acc_mean : {}".format(cur_epoch+1, recall, precision, f1_score, valid_acc))
  

  SAVE_PATH = './news_model/checkpoint/cur_epcoh_%.2f_valid_recall_%.2f_valid_precision_%.2f_valid_f1_score_%.2f__valid_acc_mea_%.2f' % (cur_epoch+1, recall, precision, f1_score, cur_acc/len(valid_dataloader))
  torch.save({'cur_epoch' : cur_epoch, 'hyper_params' : hyper_params, 'valid_recall' : recall,\
  'valid_precision':precision, 'valid_f1_score' : f1_score, 'valid_acc' : valid_acc,\
  'model_state_dict' : model.state_dict(), 'optim_state_dict' : optimizer.state_dict()},SAVE_PATH)
  

