import torch 
from torch.utils.data import Dataset
import numpy as np

class NewsDataset(Dataset) : 
  PAD = 1

  def __init__(self, input_data, label, tokenizer, max_len=512, batch_size=32) : 
    super(NewsDataset, self).__init__()

    self.batch_size = batch_size
    
    ## 1. tokenize data
    data_tokens = [ tokenizer.tokenize(cur_data) for cur_data in input_data]
    data_token_ids = [ tokenizer.convert_tokens_to_ids(cur_token)\
                           for cur_token in data_tokens]
    data_token_ids = [  cur_token_ids if len(cur_token_ids) < max_len else cur_token_ids[:max_len]\
                           for cur_token_ids in data_token_ids]

    ## 2. label
    label = [ float(cur_label) for cur_label in label]         


    ## 3. 길이 순 정렬 
    to_sort = [ (len(cur_data), cur_data, cur_label) for cur_data, cur_label in zip(data_token_ids, label)]
    to_sort.sort()

    ## 4. 저장 
    self.data_token_ids = [ cur_token_id for _,cur_token_id,_ in to_sort]
    self.valid_length = [np.int32(len(cur_token_id)) for cur_token_id in self.data_token_ids]
    self.label = [ np.float32(label) for _,_,label in to_sort]

    ## 5. pad 추가 
    self.add_pad(tokenizer)

    ## 6. set segmentid 
    self.segment_ids = [ np.int32([1]*len(cur_token_ids)) for cur_token_ids in self.data_token_ids]
    
  def __getitem__(self, idx) : 
    return (self.data_token_ids[idx], self.valid_length[idx], self.segment_ids[idx], self.label[idx])
  
  def __len__(self) : 
    return len(self.data_token_ids)

  def add_pad(self, tokenizer) : 
    PAD = tokenizer.convert_tokens_to_ids('[PAD]')
    for i in range(len(self) // self.batch_size) :
      if (i+1)*self.batch_size > len(self) :
        batch_token_ids = self.data_token_ids[i*self.batch_size : ]
      else : 
        batch_token_ids = self.data_token_ids[i*self.batch_size : (i+1)*self.batch_size]

      max_len = max([ len(cur_token_ids) for cur_token_ids in batch_token_ids ])
      for j in range(i*self.batch_size, (i+1)*self.batch_size) : 
        if j > len(self) : break 
        self.data_token_ids[j] =  torch.tensor(self.data_token_ids[j] + [PAD] * (max_len - len(self.data_token_ids[j]))) 

    return 