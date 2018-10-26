'''
train custom word2vec model based on given data
only used once
save results accordingly
'''

from gensim import utils
from gensim.models import Word2Vec
from util import load_data

# get training data
print('Loading data.')
documents = load_data(1)
# train model
print('Training word2vec.')
# print(documents[0])
model = Word2Vec(documents, min_count=1)
# summary vocabulary
words = list(model.wv.vocab)
print('Number of words')
print(len(words))
val = model['d6b72e591b91']
print(len(val))
model.save('.\\Data\\word2vec.bin')
print('word2vec model saved.')
