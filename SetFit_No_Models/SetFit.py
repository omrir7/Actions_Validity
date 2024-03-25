import pickle
from setfit import SetFitModel, SetFitTrainer
from sentence_transformers.losses import CosineSimilarityLoss

#File Content:
# 1.    Loading Datasets
# 2.    Load SetFit model and creating a trainer
# 3.    Train and Evaluate


#------------------------------------ 1 Loading Dataset (Balanced) and splitting----------------------------------
#-----------------------------------------------------------------------------------------------------------------
# with open("SetFit_Prepared_Data/train_dataset_setfit.pickle", 'rb') as file:
#     train_dataset = pickle.load(file)
# with open("SetFit_Prepared_Data/test_dataset_setfit.pickle", 'rb') as file:
#     test_dataset = pickle.load(file)
# with open("SetFit_Prepared_Data/val_dataset_setfit.pickle", 'rb') as file:
#     val_dataset = pickle.load(file)

with open("SetFit_Prepared_Data/train_data_event.pkl", 'rb') as file:
    train_dataset = pickle.load(file)
with open("SetFit_Prepared_Data/test_data_event.pkl", 'rb') as file:
    test_dataset = pickle.load(file)
with open("SetFit_Prepared_Data/eval_data_event.pkl", 'rb') as file:
    val_dataset = pickle.load(file)
#------------------------------------2 Load SetFit model and creating a trainer ----------------------------------
#-----------------------------------------------------------------------------------------------------------------
model_id = "sentence-transformers/all-mpnet-base-v2"
model = SetFitModel.from_pretrained(model_id)

# Create trainer
trainer = SetFitTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    loss_class=CosineSimilarityLoss,
    metric="accuracy",
    column_mapping={"Processed_Text": "text", "Label": "label"}, #map to my columns names
    batch_size=64,
    num_iterations=20, # The number of text pairs to generate for contrastive learning
    num_epochs=3, # The number of epochs to use for constrastive learning
)


#------------------------------------ 3 Train and Evaluate -------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------
trainer.train()
metrics = trainer.evaluate()

print(f"model used: {model_id}")
print(f"train dataset: {len(train_dataset)} samples")
print(f"accuracy: {metrics['accuracy']}")
path = 'Trained_Models/setfit_event_3'
model.save_pretrained(path)