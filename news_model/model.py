import torch
import torch.nn as nn
from torch.nn import Linear

class BERTNewsClassifier(nn.Module) : 

  def __init__(self, bert, hidden_size=768, max_len=512, num_classes=2) :
    super(BERTNewsClassifier,self).__init__()
    self.hidden_size = hidden_size

    ## bert 
    self.bert = bert

    ## fc layer 
    self.fc = Linear(hidden_size, hidden_size)

    ## classifier - logistic regression 
    self.transform = Linear(hidden_size, 2)
    self.classifier = nn.LogSoftmax(-1)

  def make_attention_mask(self, token_ids, valid_length) : 
    attention_mask = torch.zeros_like(token_ids)
    for i, cur_length in enumerate(valid_length) : 
      attention_mask[i][:cur_length] = 1
    return attention_mask.float()

  def forward(self, token_ids, valid_length, segment_ids):
    attention_mask = self.make_attention_mask(token_ids, valid_length)
    
    embed_text = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(), attention_mask= attention_mask.float().to(token_ids.device))


    h_vec = self.fc(embed_text['pooler_output'])

    h = self.transform(h_vec)
    log_probs = self.classifier(h)

    return log_probs, h_vec 