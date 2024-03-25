# Actions_Validity
With a dataset comprising 1118 transcripts, each tagged with action validity, my objective was to develop a system capable of accurately identifying and validating action occurrence. 


# Project Structure:

-   Full_System/
        Aux.py                                          auxilary functions
        PpeLine.py                                      the full system pipeline.
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
