from keras.models import Model
from keras.layers import Dense, Dropout, Input, LSTM, Bidirectional, Masking, Embedding, Concatenate, Reshape, Conv2D, MaxPool2D, Flatten
from keras.layers import BatchNormalization, Activation
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from util import get_data
import pandas as pd
import numpy as np
import random

# load data
print('Loading data...')
train_data, train_label, test_data, test_label, embed_matrix, dic, unique_label = get_data()
print('train data shape: ', len(train_data))
print('train label shape: ', len(train_label))
print('test data shape:', len(test_data))
print('test label shape: ', len(test_label))

print(unique_label)

# train_data = np.array(train_data)
train_label = np.array(train_label)
# test_data = np.array(test_data)
test_label = np.array(test_label)

sequence_length = 350   # max length after padding
vocab_size = len(dic)
embedding_dim = 100
filter_sizes = [3, 4, 5]
num_filters = 256
drop = 0.5
epochs = 10
batch_size = 256
learning_rate = 0.0001
num_classes = 14

# cnn classifier
inputs = Input(shape=(sequence_length,))
# embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=sequence_length)(inputs)
embedding = Embedding(vocab_size + 1, embedding_dim, weights=[embed_matrix], trainable=True)(inputs)
reshape = Reshape((sequence_length,embedding_dim,1))(embedding)

conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)

maxpool_0 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
maxpool_1 = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
maxpool_2 = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)

concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
flatten = Flatten()(concatenated_tensor)
dropout = Dropout(drop)(flatten)
output = Dense(units=num_classes, activation='softmax')(dropout)

model = Model(inputs=inputs, outputs=output)

checkpoint = ModelCheckpoint('.\\Weights\\weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
print("Traning Model...")
model.fit(train_data, train_label, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[checkpoint], validation_data=(test_data, test_label))  # starts training

truth = []
input = []
for i in range(5):
    j = random.randint(145, 986)
    input.append(test_data[j])
    truth.append(test_label[j])

input = np.array(input)
prediction = model.predict(input)

index = []
for l in truth:
    for i, val in enumerate(l):
        if val == 1.:
            index.append(unique_label[i])

result = []
for l in prediction:
    l = list(l)
    maxIndex = l.index(max(l))
    result.append(unique_label[maxIndex])

print("Truth: ", index)
print("Prediction: ", result)
