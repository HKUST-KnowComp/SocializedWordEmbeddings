# Socialized Word Embeddings
---------------
## Preparation
You need to download the dataset and some tools:
* Get Yelp dataset:
https://www.yelp.com/dataset_challenge/dataset

* Extract data
Get ‘json_to_csv_converter’ at https://github.com/Yelp/dataset-examples
Convert the following dataset from json format to csv format:
yelp_academic_dataset_review.json
yelp_academic_dataset_user.json

* SVM tool:
https://github.com/cjlin1/liblinear
After downloading liblinear, you can refer to “Installation” to install SVM.
It is suggested that you put liblinear under the directory SocializedWordEmbeddings, otherwise, you have to change the default directory(SocializedWordEmbeddings/liblinear) in SocializedWordEmbeddings/sentiment/sentiment.py.

* Stanford CoreNLP 
https://github.com/stanfordnlp/CoreNLP
Only stanford-corenlp.jar is required. SocializedWordEmbeddings/preprocess/Split_NN.jar and SocializedWordEmbeddings/preprocess/Split_PPL.jar need to reference stanford-corenlp.jar. 
It is suggested that after getting stanford-corenlp.jar, you put it under the directory SocializedWordEmbeddings/resources, otherwise, you should modify the default Class-Path to the path that contains stanford-corenlp.jar)

---------------
## Sentiment and Perplexity
Modify run.sh by specifying some arguments.
---------------
## User Vector for Attention
We thank Tao Lei as our code is developed based on [his code](https://github.com/taolei87/rcnn/tree/master/code).
You can simply re-implement our results of different settings (Table 5 in the paper) by modifying the run.sh: 

[1] add user and word embeddings by specifying '--user_embs' and '--embedding'

[2] add train/dev/test files by specifying '--train', '--dev', and '--test' respectively.

[3] three settings for our experiments could be achieved by specifying '--user_atten' and '--user_atten_base':

    setting '--user_atten 0' for 'Without attention'.
    
    setting '--user_atten 1 --user_atten_base 1' for 'Trained attention'
    
    setting '--user_atten 1 --user_atten_base 0' for 'Fixed user vector as attention'.
---------------
## Dependencies
[Theano](http://deeplearning.net/software/theano/) >= 0.7 
Python 2.7 
[Numpy](http://www.numpy.org) 
[Gensim](https://radimrehurek.com/gensim/install.html)
[PrettyTable](https://pypi.python.org/pypi/PrettyTable)
