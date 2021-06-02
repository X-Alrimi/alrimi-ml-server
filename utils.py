def get_f1_score(test_result) : 
  TP = 0; FP = 0; FN = 0; TN = 0
  for cur_pred, cur_label in test_result : 
    if cur_pred == cur_label : 
      if cur_pred == 0 :
        TN += 1
      else :
        TP += 1
    else : 
      if cur_pred == 0 :
        FN += 1
      else : 
        FP += 1

  recall = (TP+0.001)/(TP+FN+0.001)
  precision = (TP+0.001)/(TP+FP+0.001)
  f1_score = 2*recall*precision/(recall + precision)

  return recall, precision, f1_score
  
  
