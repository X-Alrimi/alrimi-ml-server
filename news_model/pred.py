import torch 
import torch.nn as nn
from torch.utils.data import DataLoader 
import numpy as np

from kobert_transformers import get_tokenizer, get_kobert_model 

from model import BERTNewsClassifier
from news_dataset import NewsDataset


import pandas as pd 
import numpy as np

class Pred() : 
    
    def __init__(self, checkpoint) : 
        super().__init__()
        
        bert_model = get_kobert_model()
        model =BERTNewsClassifier(bert_model, num_classes=checkpoint['hyper_params']['num_classes']).to(checkpoint['hyper_params']['device'])
        model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model = model
        self.device = checkpoint['hyper_params']['device']
        
    def __call__(self, test_dataloader) :
        outs = list();vecs = list()
        
        self.model.eval()
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
            with torch.no_grad() :
                if type(token_ids) == list : 
                    token_ids = torch.stack(token_ids).transpose(0,1).contiguous()
                if token_ids.dim() == 1 : 
                    token_ids = token_ids.unsqueeze(0)
                token_ids = token_ids.long().to(self.device)
                segment_ids = segment_ids.long().to(self.device)
                valid_length= valid_length
                label = label.long().to(self.device)
            
                out, vec = self.model(token_ids, valid_length, segment_ids)
                preds = [ int(torch.argmax(cur_out)) for cur_out in out]
            
                outs.extend([ cur_pred for cur_pred in preds])
                vecs.extend([ cur_vec for cur_vec in vec])

        return outs, vecs
        
        
if __name__ == "__main__":
    # execute only if run as a script
    LOAD_PATH = './news_model/checkpoint/cur_epcoh_8.00_valid_recall_0.99_valid_precision_0.91_valid_f1_score_0.94__valid_acc_mea_0.95'
    checkpoint = torch.load(LOAD_PATH)
    tokenizer = get_tokenizer()
    
    test_data = pd.read_csv('./data/test_data.csv')
    test_dataset = NewsDataset(test_data['input'], test_data['label'], tokenizer,\
                                checkpoint['hyper_params']['embed_size'], \
                                checkpoint['hyper_params']['batch_size'])
    
    test_dataloader = DataLoader(test_dataset, batch_size=checkpoint['hyper_params']['batch_size'],\
                                num_workers=checkpoint['hyper_params']['batch_size'])
    
    pred_module = Pred(checkpoint)
    preds, vecs = pred_module(test_dataloader)
    
    print(preds)
    print(vecs)
 