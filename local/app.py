from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import os
import numpy as np
import json

# import HashingVectorizer from local dir
# from vectorizer import vect
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Embedding, Concatenate, Reshape, Conv2D, MaxPool2D, Flatten
from keras.preprocessing import sequence

sequence_length = 350
embedding_dim =100
num_filters = 256
filter_sizes = [3,4,5]
drop = 0.5
num_classes = 14
vocab_size = 1037934

app = Flask(__name__)

######## Preparing the Classifier
def build_model():
    inputs = Input(shape=(sequence_length,))
    # embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=sequence_length)(inputs)
    embedding = Embedding(vocab_size + 1, embedding_dim, embeddings_initializer='uniform', trainable=True)(inputs)
    reshape = Reshape((sequence_length, embedding_dim, 1))(embedding)

    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid',
                    kernel_initializer='normal', activation='relu')(reshape)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid',
                    kernel_initializer='normal', activation='relu')(reshape)
    conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid',
                    kernel_initializer='normal', activation='relu')(reshape)

    maxpool_0 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1, 1), padding='valid')(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1, 1), padding='valid')(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1, 1), padding='valid')(conv_2)

    concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(drop)(flatten)
    output = Dense(units=num_classes, activation='softmax')(dropout)

    model = Model(inputs=inputs, outputs=output)
    model.load_weights('weights-9776.hdf5')
    model._make_predict_function()
    return model


cur_dir = os.path.dirname(__file__)
# clf = pickle.load(open(os.path.join(cur_dir, 'pkl_objects', 'classifier.pkl'), 'rb'))
clf = build_model()
db = os.path.join(cur_dir, 'reviews.sqlite')


def process_data(raw):
    # load dictionary
    with open('dictionary.txt', 'r') as infile:
        data = infile.read()
        dic = json.loads(data)
    # convert string input to numeric one-hot encoding
    res = []
    for word in raw.split():
        index = 0
        if word in dic:
            index = dic[word]
        res.append(index)
    text_data = [res]
    # text_data.append(res)
    text_data = sequence.pad_sequences(text_data, sequence_length)
    return text_data


def classify(document):
    label = {0: 'negative', 1: 'positive'}
    # X = vect.transform([document])
    X = process_data(document)
    y = clf.predict(X) # [0]

    unique_label = ['APPLICATION', 'BILL', 'BILL BINDER', 'BINDER', 'CANCELLATION NOTICE',
                    'CHANGE ENDORSEMENT', 'DECLARATION', 'DELETION OF INTEREST',
                    'EXPIRATION NOTICE', 'INTENT TO CANCEL NOTICE', 'NON-RENEWAL NOTICE',
                    'POLICY CHANGE', 'REINSTATEMENT NOTICE', 'RETURNED CHECK']
    result = []
    for l in y:
        l = list(l)
        maxIndex = l.index(max(l))
        result.append(unique_label[maxIndex])

    # modified
    first = y[0]
    topThree = sorted(zip(first, unique_label), reverse=True)[:3]

    return topThree
    # return y


# def sqlite_entry(path, document, y):
#     conn = sqlite3.connect(path)
#     c = conn.cursor()
#     c.execute("INSERT INTO review_db (review, sentiment, date)"\
#     " VALUES (?, ?, DATETIME('now'))", (document, y))
#     conn.commit()
#     conn.close()


######## Flask
'''
class ReviewForm(Form):
    moviereview = TextAreaField('',
                                [validators.DataRequired(),
                                validators.length(min=15)])
'''
class DocumentForm(Form):
    document = TextAreaField('', [validators.DataRequired(), validators.length(min=15)])


@app.route('/')
def index():
    #form = ReviewForm(request.form)
    form = DocumentForm(request.form)
    return render_template('documentin.html', form=form)


@app.route('/results', methods=['POST'])
def results():
    # form = ReviewForm(request.form)
    form = DocumentForm(request.form)
    if request.method == 'POST' and form.validate():
        review = request.form['document']
        # y, proba = classify(review)
        y = classify(review)
        # return render_template('results.html', content=review, prediction=y, probability=round(.05 * 100, 2))     modified
        return render_template('results.html', prediction=y)
    return render_template('documentin.html', form=form)


if __name__ == '__main__':
    app.run(debug=True)
