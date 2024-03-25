from Aux import params_list,prepare_bert_input, clean_and_lemmatize, find_matched_params_inference
from transformers import DistilBertForSequenceClassification
import torch
from setfit import SetFitModel



#File Content:
# 1.    action_detector() - Applies the full pipeline of my system.
#       A. Preprocessing and Finding actions in the transcript
#       B. Bert Section - if selected Bert for inference, convert the input to bert format and infer
#       C. SetFit Section - The input not converted and just pushed into the trained setfit model
# 2.    Applying the pipeline on a transcript



#------------------- 1 action_detector() - Applies the full pipeline of my system. ------------
#----------------------------------------------------------------------------------------------


def action_detector(transcript,Models_Path_Bert,Models_Path_SetFit,best_model_bert,best_model_setfit,selected_model):

# ------------------- 1.A Preprocessing and Finding actions in the transcript ----- ------------
# ----------------------------------------------------------------------------------------------
    processed_input = clean_and_lemmatize(transcript)
    actions = find_matched_params_inference(processed_input,params_list)

    if not actions:
        return []
    else:
# ----------------------------------- 1.B Bert Section-----------------------------------------
# ----------------------------------------------------------------------------------------------
        if selected_model=='bert':
            best_model_path = Models_Path_Bert+best_model_bert
            model = DistilBertForSequenceClassification.from_pretrained(best_model_path)
            bert_formated_text = prepare_bert_input(text=transcript, matched_params=params_list)
            input_ids = torch.tensor(bert_formated_text['input_ids'])
            attention_mask = torch.tensor(bert_formated_text['attention_mask'])
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.sigmoid(logits)
            predicted_labels = torch.round(probabilities)
            if (predicted_labels[0][0]==1): #if model predicted 0 - return also the correctness probability
                print(f'Not A Valid Action')
                return [], float(probabilities[0][0])
            else:                           #if model predicted 1
                print(f'Valid Action: {actions}, prob = {float(probabilities[0][1])}')
                return actions, float(probabilities[0][1])

# ----------------------------------- 1.C SetFit Section-----------------------------------------
# -----------------------------------------------------------------------------------------------
        else:
            best_model_path = Models_Path_SetFit+best_model_setfit
            best_model = SetFitModel.from_pretrained(best_model_path)
            output = best_model(processed_input)
            if(output==1):
                print(f'Valid Action: {actions}')
                return actions
            else:
                print(f'Not A Valid Action')
                return []



# ---------------------------- 2 Applying the pipeline on a transcript---------------------------
# -----------------------------------------------------------------------------------------------

#Input Transcript
in_tran = "Missed Shot by Darren CollisonRebound by Joel Embiid	If you go into that defensive circle and post up, you notice the defensive players behind you and policy and left that jump look a little bit short."
selected_model = 'setfit'


#Best Models Configuration (According to performance on testset)
best_model_bert = 'distilbert-base-uncased_40_lr4e-05_drop0.5_step5_pat3_batch55.pth'
Models_Path_Bert = '../Bert/Trained_Models/'
best_model_setfit = 'setfit_event_0'
Models_Path_SetFit = '../SetFit/Trained_Models/'

#Run action_detector()
if selected_model=='setfit':
    actions = action_detector(in_tran,Models_Path_Bert, Models_Path_SetFit,best_model_bert,best_model_setfit,selected_model=selected_model)
else:
    actions, validity_probability = action_detector(in_tran,Models_Path_Bert, Models_Path_SetFit,best_model_bert,best_model_setfit,selected_model=selected_model)

