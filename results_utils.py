import pandas as pd
import os
import numpy as np
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score
import cv2 as cv

# # Compute the average segmentation accuracy across all classes
# def compute_global_accuracy(pred, label):
#     total = len(label)
#     count = 0.0
#     for i in range(total):
#         if pred[i] == label[i]:
#             count = count + 1.0
#     return float(count) / float(total)

# # Compute the class-specific segmentation accuracy
# def compute_class_accuracies(pred, label, num_classes):
#     total = []
#     for val in range(num_classes):
#         total.append((label == val).sum())

#     count = [0.0] * num_classes
#     for i in range(len(label)):
#         if pred[i] == label[i]:
#             count[int(pred[i])] = count[int(pred[i])] + 1.0

#     # If there are no pixels from a certain class in the GT, 
#     # it returns NAN because of divide by zero
#     # Replace the nans with a 1.0.
#     accuracies = []
#     for i in range(len(total)):
#         if total[i] == 0:
#             accuracies.append(1.0)
#         else:
#             accuracies.append(count[i] / total[i])

#     return accuracies


# def compute_mean_iou(pred, label):

#     unique_labels = np.unique(label)
#     num_unique_labels = len(unique_labels)

#     I = np.zeros(num_unique_labels)
#     U = np.zeros(num_unique_labels)

#     for index, val in enumerate(unique_labels):
#         pred_i = pred == val
#         label_i = label == val

#         I[index] = float(np.sum(np.logical_and(label_i, pred_i)))
#         U[index] = float(np.sum(np.logical_or(label_i, pred_i)))


#     mean_iou = np.mean(I / U)
#     return mean_iou


def evaluate_segmentation(pred, label):
    flat_pred = pred.flatten()
    flat_label = label.flatten()

    return compute_confusion_matrix(flat_pred, flat_label)

def compute_confusion_matrix(pred, label):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    for i in range(len(label)):
        if pred[i] == label[i]:
            if pred[i] == 1:
                tp += 1
            else:
                tn += 1
        else:
            if pred[i] == 1:
                fp += 1
            else:
                fn += 1
    
    # print("Confusion Matrix : ")
    # print(f"[{tp}] [{fp}]")
    # print(f"[{fn}] [{tn}]")
    return tp, tn, fp, fn


def prepare_data(dataset_dir):

    gt_names=[]
    pred_names=[]

    for file in os.listdir(dataset_dir):
        cwd = os.getcwd()
        if '_gt' in file:
            gt_names.append(cwd + "/" + dataset_dir + '/' + file)
        elif '_pred' in file:
            pred_names.append(cwd + "/" + dataset_dir + '/' + file)
    gt_names.sort(), pred_names.sort()
    return gt_names, pred_names

def load_image(path):
    image = cv.cvtColor(cv.imread(path,-1), cv.COLOR_BGR2RGB)
    return image

def filepath_to_name(full_name):
    file_name = os.path.basename(full_name)
    file_name = os.path.splitext(file_name)[0]
    return file_name


def calc_metrics_from_conf(tp, fp, fn, tn):
    acc = (tp+tn)/(tp+fp+fn+tn)
    prec = tp/(tp + fp)
    recall = tp/(tp + fn)
    # f1 = 2*tp/(2*tp + fp + fn)
    f1 = (2*prec*recall)/(prec + recall)
    iou_sunlit = tp/(tp + fp + fn)
    iou_noise = tn/(tn + fn + fp)
    mean_iou = (iou_sunlit + iou_noise)/2
    return acc, prec, recall, f1, mean_iou