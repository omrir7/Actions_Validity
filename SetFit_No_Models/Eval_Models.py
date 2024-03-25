from setfit import SetFitModel
import pickle
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os


#File Content:
# 1.    Loading TestSet
# 2.    Iterating over the Trained Models check their performance on the testset
#       (calc acc, f1 and plotting confusion matrices.


#------------------------------------ 1 Loading TestSet ----------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------
#path = "SetFit_Prepared_Data/test_dataset_setfit.pickle"
path = "SetFit_Prepared_Data/test_data_event.pkl"

with open(path, 'rb') as file:
    test = pickle.load(file)
texts = list(test['Processed_Text'])
labels = list(test['Label'])



#------------------------------------ 1 Iterating over the Trained Models ----------------------------------------
#-----------------------------------------------------------------------------------------------------------------

f1_scores = []
acc_scores = []
names = []
max_f1 = 0
max_acc = 0
max_f1_idx = 0
max_acc_idx = 0
i=0
models_directory = 'Trained_Models'
import numpy as np

# Iterate over each file in the directory
for filename in os.listdir(models_directory):
    # Load the model
    model_path = os.path.join(models_directory, filename)
    names.append(model_path)
    best_model = SetFitModel.from_pretrained(model_path)

    # Inference
    predictions = best_model(texts)  # Predict
    predictions = list(predictions)
    count_same = sum(x == y for x, y in zip(predictions, labels))
    acc = count_same / len(labels)
    acc_scores.append(acc)

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(labels, predictions)

    # Calculate precision, recall, accuracy, and F1-score
    precision = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])
    recall = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])
    accuracy = (conf_matrix[0, 0] + conf_matrix[1, 1]) / np.sum(conf_matrix)
    f1 = 2 * (precision * recall) / (precision + recall)

    # Plot the confusion matrix with annotations
    # Plot the confusion matrix with annotations
    plt.figure(figsize=(12, 8))  # Increase image width
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])

    # Annotate precision, recall, accuracy, and F1-score on top right
    plt.text(2.7, 2, f'Precision: {precision:.2f}\nRecall: {recall:.2f}\nAccuracy: {accuracy:.2f}\nF1-score: {f1:.2f}',
             horizontalalignment='right', verticalalignment='top', fontsize=12)

    plt.xlabel('Predicted Valid')
    plt.ylabel('Actual Valid')
    plt.title(f'Confusion Matrix for Model: {filename}')  # Set title with model name
    plt.show()

    # Update best performing models
    if f1 > max_f1:
        max_f1 = f1
        max_f1_idx = i
    if acc > max_acc:
        max_acc = acc
        max_acc_idx = i
    i += 1

# Print results
for i in range(len(names)):
    print(f'{names[i]} f1: {f1_scores[i]}  acc: {acc_scores[i]}')
print('')
print(f'Best Acc for model: {names[max_acc_idx]} - {max_acc}')
print(f'Best F1 for model: {names[max_f1_idx]}  -  {max_f1}')
