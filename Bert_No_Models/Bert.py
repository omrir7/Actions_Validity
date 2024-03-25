from transformers import DistilBertTokenizer
import pickle
from sklearn.model_selection import train_test_split
from transformers import DistilBertForSequenceClassification, AdamW
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import itertools

#File Content:
# 1.    Preparing the dataset for bert -
#           A. Loading Dataset
#           B. Converting to Bert Format: "[CLS] ... transcript tokens ... [SEP] ..actions..[SEP]",
#           C. Splitting the Dataset to Train,Val,Test While keeping balancing.
# 2.    Training Function
# 3.    Hyperparameters Search and Training
#           A. Defining the search space.
#           B. Iterating over the Hyperparameters space, training model instances and evaluating them.

#------------------------------------ 1.A Loading The Dataset ---------------------------------
#----------------------------------------------------------------------------------------------

balanced_path = "../Preprocessed_Data/balanced_dataset.pickle"
with open(balanced_path, 'rb') as file:
    balanced_dataset = pickle.load(file)

# Load the pre-trained DidtilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

#------------------------------------ 1.B Converting to Bert Format ---------------------------------
#----------------------------------------------------------------------------------------------------

# [CLS] ... transcript tokens ... [SEP] ..actions..[SEP]
def prepare_bert_input(text, matched_params, max_seq_length=128):
    text_tokens = tokenizer.tokenize(text)
    param_tokens = tokenizer.tokenize(' '.join(matched_params))

    max_text_length = (max_seq_length - 3) // 2  # 3 (2 for [CLS] and [SEP], 1 for [SEP] between text and params)
    text_tokens = text_tokens[:max_text_length]
    param_tokens = param_tokens[:max_seq_length - len(text_tokens) - 2]  # Account for [CLS], [SEP], and text tokens

    # [CLS] ... transcript tokens ... [SEP] ..actions..[SEP]
    input_tokens = ['[CLS]'] + text_tokens + ['[SEP]'] + param_tokens + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)

    attention_mask = [1] * len(input_ids)  # (1 for real tokens, 0 for padding tokens)
    # Pad sequences to max_seq_length
    padding_length = max_seq_length - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * padding_length
    attention_mask += [0] * padding_length
    return {'input_ids': input_ids, 'attention_mask': attention_mask}


# convert input to Bert Format
inputs = balanced_dataset.apply(lambda row: prepare_bert_input(row['Processed_Text'], row['Matched_Params']), axis=1)

# Convert input sequences to tensors
input_ids = torch.tensor([x['input_ids'] for x in inputs])
attention_mask = torch.tensor([x['attention_mask'] for x in inputs])
labels = torch.tensor(balanced_dataset['Label'].values)

# Convert inputs and labels to lists
input_list = inputs.tolist()
label_list = labels.tolist()

##------------------1.C Splitting the Dataset to Train,Val,Test While keeping balancing. ---------------------------------
##------------------------------------------------------------------------------------------------------------------------


# Split the dataset into training, validation, and testing sets (80% train, 10% validation, 10% test)
train_inputs, temp_inputs, train_labels, temp_labels = train_test_split(input_list, label_list, test_size=0.2, random_state=42, stratify=label_list)
val_inputs, test_inputs, val_labels, test_labels = train_test_split(temp_inputs, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels)

# Convert input sequences to PyTorch tensors
train_input_ids = torch.tensor([x['input_ids'] for x in train_inputs])
train_attention_mask = torch.tensor([x['attention_mask'] for x in train_inputs])

# Convert labels to one-hot encoding
train_labels_onehot = torch.eye(2)[train_labels]

val_input_ids = torch.tensor([x['input_ids'] for x in val_inputs])
val_attention_mask = torch.tensor([x['attention_mask'] for x in val_inputs])

# Convert labels to one-hot encoding
val_labels_onehot = torch.eye(2)[val_labels]

test_input_ids = torch.tensor([x['input_ids'] for x in test_inputs])
test_attention_mask = torch.tensor([x['attention_mask'] for x in test_inputs])

# Convert labels to one-hot encoding
test_labels_onehot = torch.eye(2)[test_labels]

# Create PyTorch datasets
train_dataset = torch.utils.data.TensorDataset(train_input_ids, train_attention_mask, train_labels_onehot)
val_dataset = torch.utils.data.TensorDataset(val_input_ids, val_attention_mask, val_labels_onehot)
test_dataset = torch.utils.data.TensorDataset(test_input_ids, test_attention_mask, test_labels_onehot)

print(f'Train Samples: {len(train_dataset)}')
print(f'Validation Samples: {len(val_dataset)}')
print(f'Test Samples: {len(test_dataset)}')


# Count the occurrences of each class label in the training dataset
train_class_counts = {}
for label in train_labels:
    if label in train_class_counts:
        train_class_counts[label] += 1
    else:
        train_class_counts[label] = 1

# Count the occurrences of each class label in the validation dataset
val_class_counts = {}
for label in val_labels:
    if label in val_class_counts:
        val_class_counts[label] += 1
    else:
        val_class_counts[label] = 1

# Count the occurrences of each class label in the test dataset
test_class_counts = {}
for label in test_labels:
    if label in test_class_counts:
        test_class_counts[label] += 1
    else:
        test_class_counts[label] = 1

print("Class distribution in training set:")
print(train_class_counts)

print("\nClass distribution in validation set:")
print(val_class_counts)

print("\nClass distribution in test set:")
print(test_class_counts)

#------------------------------------------2 Training Function -----------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def train(model, train_loader, val_loader, optimizer, scheduler, num_epochs, dropout_val, patience=5):
    model.to(device)
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    dropout = nn.Dropout(dropout_val)  # Apply dropout to BERT model
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        total_loss = 0.0
        for i, batch in enumerate(train_loader):
            input_ids, attention_mask, labels = batch #pulling the ids, masks and labels from the dataloader
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask) #pass it through the model
            logits = outputs.logits #output value of the model before softmax
            logits = dropout(logits)  # Apply dropout
            loss = criterion(logits, labels) #calculating loss (BCE)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        # Compute validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_batch in val_loader:
                val_input_ids, val_attention_mask, val_labels = val_batch
                val_input_ids, val_attention_mask, val_labels = val_input_ids.to(device), val_attention_mask.to(device), val_labels.to(device)
                val_outputs = model(val_input_ids, attention_mask=val_attention_mask)
                val_logits = val_outputs.logits
                val_loss += criterion(val_logits, val_labels).item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        train_losses.append(total_loss/len(train_loader))
        scheduler.step() #changing lr if needed
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {total_loss/len(train_loader)}, Validation Loss: {val_loss}')

        # Check for early stopping - if lr not improving for <patient> epochs -> early stop
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    # Plot the losses
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.show()





#------------------------------------------3 Hyperparameters Search and Training -----------------------------------
#-------------------------------------------------------------------------------------------------------------------

#------------------------------------------3.A Defining the search space. ------------------------------------------
#-------------------------------------------------------------------------------------------------------------------
model_id = 'distilbert-base-uncased'
scheduler_step = 5
epochs_list = [20,60]
patience_list = [10,5]
dropout_list = [0.6]
lr_list = [3e-5,7e-6,2e-5,4e-5]
batch_size_list = [55,64,80]

#-----------3.B. Iterating over the Hyperparameters space, training model instances and evaluating them. -----------
#-------------------------------------------------------------------------------------------------------------------

permutations = itertools.product(epochs_list, patience_list, dropout_list, lr_list, batch_size_list) #generate all possible permutations
perm_idx=0
for epochs, patience, dropout, lr, batch_size in permutations:
    print(f'perm_idx: {perm_idx}')
    print(f"epochs: {epochs}, patience: {patience}, dropout: {dropout}, lr: {lr}, batch_size: {batch_size}")

    train_loader_balanced = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) #needed here because batch_size is changing
    val_loader_balanced = DataLoader(val_dataset, batch_size=batch_size, shuffle=True) #needed here because batch_size is changing
    test_loader_balanced = DataLoader(test_dataset, batch_size=batch_size) #needed here because batch_size is changing

    model = DistilBertForSequenceClassification.from_pretrained(model_id, num_labels=2)
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=0.1)
    train(model, train_loader_balanced, val_loader_balanced, optimizer, scheduler, num_epochs=epochs, patience=patience, dropout_val=dropout)
    # Save the trained model
    path =     model_path = f"Trained_Models/{model_id}_{epochs}_lr{lr}_drop{dropout}_step{scheduler_step}_pat{patience}_batch{batch_size}.pth"
    model.save_pretrained(model_path)
    perm_idx+=1