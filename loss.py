import torch
import torch.nn as nn

class LabelSmoothingLoss(nn.Module) :
  def __init__(self, num_classes, smoothing=0.2) :
    super(LabelSmoothingLoss, self).__init__()
    self.num_classes = num_classes
    self.smoothing = smoothing

  def __call__(self, pred, target) :
    # | target | = (batch size, )
    # | index | = (index, 1) 

    # for one hot encoding 
    index = target.view((target.size()[0],1)).to(0)
    src = torch.ones((target.size()[0],1)).to(0)
    one_hot = torch.zeros((target.size()[0], self.num_classes)).to(0).scatter_(1, index, src)

    
    one_hot = one_hot*(1-self.smoothing) + self.smoothing/self.num_classes

    # for nll loss with float preds (pred is log likelihood)
    nll_loss = ((-pred) * one_hot).mean()

    return nll_loss
