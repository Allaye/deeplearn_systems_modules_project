Steps for the Project development

Data collection
data was collected from kaggle and the data was in csv format

Data Preparation and Preprocessing

the data was cleaned and preprocessed using python and pandas, there was no much cleaning need,
the buck of the work, was to drop columns, drop the section of the languages not needed.

Data selection reasons

undersampling data since the data is imbalanced is huge and it will be difficult to train the model
below is the sample of data we have
english: 1586621
french: 501241
yoruba: 16911
ewe: 28640



Challenges
dealing with text data is a challenging process

so after the data has been processed, " tokinized, vocab created", i created a collated_fn
that will be used to convert the vocab to tensor, but on my investigation, i found out that,
the collated function is out putting zero for some words meaning, those words are not in the vocab
which is surprising. i found out that my sentence processing pipeline used inside the collated_fn
wasnt performing the preprocessing as the function used to generate tokens for the vocab.






# Architecture Design Values
network - embedding 
        - RNN
        - LSTM



Parameter Value
Vocabulary size 300
Embedding dimension 128
Number of epochs 30
Batch size 128

Number of epochs 60
Batch size 32
Activation ReLU
Optimizer Adam
Dropouts Yes
Loss Categorical cross-entropy
Activation output Softmax
Word embedding
Number of epochs 25
CNN model Word embedding Keras
Loss Categorical cross-entropy
Optimizer Adam
Activation output Softmax
Na¨ıve Bayes
Type Bernoulli Na¨ıve Bayes
Kernel function Bernoulli



## Architecture Design Results
  >> Vanilla Embedding Model -> 0.51% accuracy /b
  >> Pretrained Embedding Model -> 0.62% accuracy
  