import os
from transformers import DistilBertForSequenceClassification
from torch.utils.data import DataLoader
import torch
import pickle


#File Content:
# 1.    Loading Dataset (Test)
# 2.    Iterating over the Trained Models and evaluating them
# 3.    Calculate Total Average Accuracy

#------------------------------------ 1 Loading The Dataset -----------------------------------------------------
#----------------------------------------------------------------------------------------------------------------

path = "Bert_Prepared_Data/test_dataset_bert.pickle"
with open(path, 'rb') as file:
    test = pickle.load(file)
test_loader_balanced = DataLoader(test, batch_size=32)


#------------------------------------ 2 Iterating over the Trained Models and evaluating them -------------------
#----------------------------------------------------------------------------------------------------------------
model_directory = "Trained_Models/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
total_accuracy = 0
num_models = 0
# Iterate over each file in the directory
for filename in os.listdir(model_directory):
    if filename.endswith(".pth"):  # Check if the file is a model file
        model_path = os.path.join(model_directory, filename)
        model = DistilBertForSequenceClassification.from_pretrained(model_path)

        model.eval()
        total_correct = 0
        # Iterate through the test set in batches
        for batch in test_loader_balanced:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            # Disable gradient calculations
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probabilities = torch.sigmoid(logits)
                predicted_labels = torch.round(probabilities) # Round probabilities to get predicted labels
                total_correct += torch.sum(torch.all(predicted_labels == labels, dim=1))

        # Calculate accuracy for the current model
        accuracy = total_correct / len(test_loader_balanced.dataset)
        print(f"Model: {filename}, Test Accuracy: {accuracy * 100:.2f}%")

        # Update total accuracy across all models
        total_accuracy += accuracy
        num_models += 1


#------------------------------------ 3 Calculate Total Average Accuracy -----------------------------------
#-----------------------------------------------------------------------------------------------------------
average_accuracy = total_accuracy / num_models
print(f"Average Test Accuracy Across All Models: {average_accuracy * 100:.2f}%")
