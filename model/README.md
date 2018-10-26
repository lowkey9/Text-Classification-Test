This folder gives the implementation of text classifier which is a Convolutional Neural Network.  
At first I trained a custom word2vec model which can be used for word embedding. 
However, the generated files are too large to be uploaded to github.  
So, I switched to using one-hot encoding of each word as the input. And let the Embedding layer learn representations of words later during training.  
Run **train_onehot.py** to check the model.
