# IPython log file

get_ipython().magic('load digit-recognizer.py')
# %load digit-recognizer.py
# IPython log file

import numpy

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder


def split_data(data):
    labels = data[:, 0]
    pixels = data[:, 1:]
    return labels, pixels

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

print("-> Loading dataset...")
dataset = numpy.loadtxt("data/train.csv", delimiter=",")
print("-> Splitting input and labels...")
labels, pixels = split_data(dataset)

encoder = LabelEncoder()
encoder.fit(labels)
encoded_labels = encoder.transform(labels)
categorical_labels = np_utils.to_categorical(encoded_labels)


print("-> Splitting train set...")
train_labels, train_pixels = categorical_labels[:33600], pixels[:33600]
print("-> Splitting test set...")
test_labels, test_pixels = categorical_labels[33600:], pixels[33600:]


model = Sequential()
model.add(Dense(32, input_dim=784, init='uniform', activation='relu'))
model.add(Dense(16, init='uniform', activation='relu'))
model.add(Dense(10, init='uniform', activation='sigmoid'))

print("-> Model created")
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print("-> Model compiled")
# Fit the model
model.fit(train_pixels, train_labels, nb_epoch=150)
print("-> Model fit")
# evaluate the model
scores = model.evaluate(train_pixels, train_labels)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
scores = model.evaluate(test_pixels, test_labels)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
get_ipython().magic('logstart experiment1.py')
help(model.predict)
categorical_labels[:10]
labels[:10]
testset = numpy.loadtxt("data/test.csv", delimiter=",")
testset
testset.shape
help(model.predict)
submission = model.predict(testset)
submission
submission.shape
dataset
pixels
type(submission)
submission[0].argmax()
lambda x = x.argmax()
val = lambda x: x.argmax()
result = [val(x) for x in submission]
result
result[:10]
submission[:10]
with f as open("./rnn_submission.csv"):
    f.write("ImageId,Label")
    count = 0
    for r in result:
        count += 1
        f.write("%s,%s", count, r)
with open("./rnn_submission.csv") as f:
    f.write("ImageId,Label")
    count = 0
    for r in result:
        count += 1
        f.write("%s,%s", count, r)
        
with open("./rnn_submission.csv", "w+") as f:
    f.write("ImageId,Label")
    count = 0
    for r in result:
        count += 1
        f.write("%s,%s", count, r)
        
with open("./rnn_submission.csv", "w+") as f:
    f.write("ImageId,Label")
    count = 0
    for r in result:
        count += 1
        f.write("%s,%s" % count, r)
        
with open("./rnn_submission.csv", "w+") as f:
    f.write("ImageId,Label")
    count = 0
    for r in result:
        count += 1
        f.write("%s,%s" % (count, r))
        
with open("./rnn_submission.csv", "w+") as f:
    f.write("ImageId,Label\n")
    count = 0
    for r in result:
        count += 1
        f.write("%s,%s\n" % (count, r))
        
get_ipython().magic('logstart rnn_submission.py')
get_ipython().magic('logstate')
