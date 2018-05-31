Unsupervised Language Learning Assignment 3    

Melissa Tjhia (10761071)    
Richard Olij (10833730)    
###########################################

There is a folder called "skipgram", in which the skip gram models are trained.     
The europarl folder should be in that folder.    
Trained models are given.
New models can be trained by running the "train.py" file.    
After training, files called "wordvec{}.text" and "word2id{}.text" are saved, where the model's information is filled in the {}.    

The folder called skipgram_data_trained and senteval_skipgram.ipynb should be in the "SentEval" folder.
Unzip the "skipgram_data_trained/word_vectors" and "skipgram_data_trained/word2id".  

Only with new trained models:     
"wordvec{}.text" should be moved to "skipgram_data_trained/word_vectors"    
"word2id{}.text" should be moved to "skipgram_data_trained/word2id"   

This is already done for the models that were trained for the report.     

In "senteval_skipgram.ipynb", the variable "model_info" should be filled in with the model's information, which is in the spot of the above mentioned {}.    
"params.wvec_dim" should also be set accordingly (for the trained models, 100 or 300).    
Run the notebook to get the results on the SentEval tasks.    

###########################################

We could not store some of the trained data on github, instead download the models and wordvectors from https://we.tl/VwpbTwvTu0 if you want to run the trained data.

The zips should be unzipped here:
skipgram/models.zip
skipgram_data_trained/word_vectors.zip

The data is available to the 6th of june 2018. Contact us otherwise if the data is not available anymore.
