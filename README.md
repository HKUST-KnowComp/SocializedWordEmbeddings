# Socialized Word Embeddings

## Preparation
You need to download the dataset and some tools:
* Download [Yelp dataset](https://www.yelp.com/dataset_challenge/dataset)

* Convert following datasets from json format to csv format by using [`json_to_csv_converter.py`](https://github.com/Yelp/dataset-examples):
  
  yelp_academic_dataset_review.json
  
  yelp_academic_dataset_user.json

* Download [LIBLINEAR](https://github.com/cjlin1/liblinear)
  
  After downloading `liblinear`, you can refer to `Installation` to install it.
  
  It is suggested that you put `liblinear` under the directory `SocializedWordEmbeddings`.

* Download [Stanford CoreNLP](https://github.com/stanfordnlp/CoreNLP)
  
  Only `stanford-corenlp.jar` is required. `SocializedWordEmbeddings/preprocess/Split_NN.jar` and    `SocializedWordEmbeddings/preprocess/Split_PPL.jar` need to reference `stanford-corenlp.jar`. 
  
  
  It is suggested that after getting `stanford-corenlp.jar`, you put it under the directory `SocializedWordEmbeddings/resources`, otherwise, you should modify the default `Class-Path` in `Split_NN.jar` and `Split_PPL.jar`.

## Preprocessing
`cd SocializedWordEmbeddings/preprocess`

Modify `./run.py` by specifying `--input` (Path to yelp dataset).

`python run.py`

## Training
`cd SocializedWordEmbeddings/train`

You may modify the following arguments in `./run.py`:

* `--para_lambda`     The trade off parameter between log-likelihood and regularization term
* `--para_r`     The constraint of L2-norm of the user vector
* `--yelp_round`     The round number of yelp data, e.g. {8,9}

`python run.py`

## Sentiment Classification
`cd SocializedWordEmbeddings/sentiment`

You may modify the following arguments in `./run.py`:

* `--para_lambda`     The trade off parameter between log-likelihood and regularization term
* `--para_r`     The constraint of L2-norm of the user vector
* `--yelp_round`     The round number of yelp data, e.g. {8,9}

`python run.py`

## Perplexity
`cd SocializedWordEmbeddings/perplexity`

You may modify the following arguments in `./run.py`:
* `--para_lambda`     The trade off parameter between log-likelihood and regularization term
* `--para_r`     The constraint of L2-norm of the user vector
* `--yelp_round`     The round number of yelp data, e.g. {8,9}

`python run.py`

## User Vectors for Attention

We thank Tao Lei as our code is developed based on [his code](https://github.com/taolei87/rcnn/tree/master/code).

You can simply re-implement our results of different settings (Table 5 in the paper) by modifying the `SocializedWordEmbeddings/attention/run.sh`: 

[1] add user and word embeddings by specifying `--user_embs` and `--embedding`.

[2] add train/dev/test files by specifying `--train`, `--dev`, and `--test` respectively.

[3] three settings for our experiments could be achieved by specifying `--user_atten` and `--user_atten_base`:

    setting '--user_atten 0' for 'Without attention'.
    
    setting '--user_atten 1 --user_atten_base 1' for 'Trained attention'
    
    setting '--user_atten 1 --user_atten_base 0' for 'Fixed user vector as attention'.

## Dependencies

* Python 2.7 
* [Theano](http://deeplearning.net/software/theano/) >= 0.7
* [Numpy](http://www.numpy.org) 
* [Gensim](https://radimrehurek.com/gensim/install.html)
* [PrettyTable](https://pypi.python.org/pypi/PrettyTable)
