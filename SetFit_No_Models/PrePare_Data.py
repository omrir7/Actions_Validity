from datasets import Dataset
from sklearn.model_selection import train_test_split
import pickle
#File Content:
# 1.    Loading Dataset (Balanced) and splitting
# 2.    Save Splitted Dataset



#------------------------------------ 1 Loading Dataset (Balanced) and splitting----------------------------------
#----------------------------------------------------------------------------------------------------------------
path = "../Preprocessed_Data/balanced_dataset.pickle"
with open(path, 'rb') as file:
    balanced_dataset = pickle.load(file)

#70,15,15
train_df, remaining_df = train_test_split(balanced_dataset, test_size=0.7, random_state=42)
eval_df, test_df = train_test_split(remaining_df, test_size=0.5, random_state=42)
# Convert DataFrames to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)
test_dataset = Dataset.from_pandas(test_df)


#------------------------------------ 2 Save Splitted Dataset----------------------------------------------------
#----------------------------------------------------------------------------------------------------------------

train_path = 'SetFit_Prepared_Data/train_dataset_setfit.pickle'
val_path = 'SetFit_Prepared_Data/val_dataset_setfit.pickle'
test_path = 'SetFit_Prepared_Data/test_dataset_setfit.pickle'

with open(train_path, 'wb') as file:
    pickle.dump(train_dataset, file)
with open(val_path, 'wb') as file:
    pickle.dump(eval_dataset, file)
with open(test_path, 'wb') as file:
    pickle.dump(test_dataset, file)