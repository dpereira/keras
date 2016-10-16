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


train_pixels = pixels
train_labels = categorical_labels

model = Sequential()
model.add(Dense(128, input_dim=784, init='uniform', activation='relu'))
model.add(Dense(64, init='uniform', activation='relu'))
model.add(Dense(10, init='uniform', activation='sigmoid'))

print("-> Model created")
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy'])
print("-> Model compiled")
# Fit the model
model.fit(train_pixels, train_labels, nb_epoch=50)
print("-> Model fit")
# evaluate the model
scores = model.evaluate(train_pixels, train_labels)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

testset = numpy.loadtxt("data/test.csv", delimiter=",")
testset
testset.shape
submission = model.predict(testset)
submission
submission.shape
dataset
pixels
type(submission)
submission[0].argmax()
val = lambda x: x.argmax()
result = [val(x) for x in submission]
result
result[:10]
submission[:10]

with open("./rnn_submission5.csv", "w+") as f:
    f.write("ImageId,Label\n")
    count = 0
    for r in result:
        count += 1
        f.write("%s,%s\n" % (count, r))
