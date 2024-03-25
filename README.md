# Actions_Validity
With a dataset comprising 1118 transcripts, each tagged with action validity, my objective was to develop a system capable of accurately identifying and validating action occurrence. 
* I haven't uploaded the the trained models due to their sizes.

  * If you want to run the full project you should:
  1. Download the required libraries from requieremnts.txt
  2. Train one of the models, you better choose SetFit.
  3. In order to train an instance of SetFit you should run SetFit.py which located in the SetFit Dir.

# System Overview

![image](https://github.com/omrir7/Actions_Validity/assets/71921802/114f78e9-106e-4360-8b0f-89d2c3bc2c8f)

# Project Structure:

-   Full_System/

        Aux.py                                          auxilary functions

        PipeLine.py                                      the full system pipeline.

    
-   Original_dataset/

        action_enrichment_ds_home_exercise.csv          the supplied dataset

        params_list.csv                                 the supplied params list

    
-   Preprocessed_Data/
 
        balanced_dataset.pickle                         the full preprocessed dataset from Google Colab (downloaded manually)
  
        test_dataset_bert.pickle                        splitted
  
        train_dataset_bert.pickle                       splitted
  
        val_dataset_bert.pickle                         splitted

  
-   Bert/
  
        Bert.py                                         Bert Training and preparing the data
    
        Bert_Prepared_Data/                             Bert data saved after processing it to match bert
    
        Trained_Models/                                 all trained models
    
        Eval_Models.py                                  script to evaluate all trained models

    
-   SetFit

        SetFit.py                                       Setfit training
    
        SetFit_Prepared_Data/                           setfit data
    
        Trained_Models/                                 setfit trained models
    
        Eval_Models.py                                  script to evaluate all trained models
    
