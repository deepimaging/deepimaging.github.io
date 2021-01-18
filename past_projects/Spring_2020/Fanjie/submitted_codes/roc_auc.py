import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize

def calculate_auc(scores_list, ground_truth_list):
    class_name = ['cls1','cls2','cls3']
    roc_dict = []
    ground_truth = []  
    scores_list = torch.cat((scores_list),dim=0)
    ground_truth_list = torch.cat((ground_truth_list),dim=0)

    assert scores_list.size(0) == ground_truth_list.size(0)
    for i, (score, label) in enumerate(zip(scores_list, ground_truth_list)):
        new_item = {}
        score = list(np.array(score))
        score.append(0)
        label = int(label)
        for idx, name in enumerate(class_name):
            new_item[name] = float(score[idx])
        
        ground_truth.append(label)
        roc_dict.append(new_item)
        
    class_length = len(class_name)
    ground_truth = label_binarize(ground_truth, classes=range(class_length))
    auc_list = []
    fpr_list = []
    tpr_list = []

    for i in range(class_length):
        prediction = []
        for item in roc_dict:
            prediction.append(item[class_name[i]])
        if (ground_truth[:,i] == 0).all():
            continue
        else:
            fpr, tpr, threshold = roc_curve(ground_truth[:,i],prediction)
            auc_list.append(auc(fpr, tpr))
    return np.array(auc_list), np.array(fpr_list), np.array(tpr_list)          
