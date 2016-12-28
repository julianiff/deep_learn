"""
Stub Code from Keras Notebook mlfa
https://github.com/ml4a/ml4a-guides/blob/master/notebooks/recurrent_neural_networks.ipynb
"""

import random
import numpy as np
from glob import glob
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional
from keras.layers.core import Dense, Activation, Dropout

# load up our text
text_files = glob('data/truob/*.txt')
text = '\n'.join([open(f, 'r').read() for f in text_files])

# extract all (unique) characters
# these are our "categories" or "labels"
chars = list(set(text))

# set a fixed vector size
# so we look at specific windows of characters
max_len = 20

model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(max_len, len(chars))))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
"""
The categorical cross-entropy loss the standard loss function for multilabel classification,
which basically penalizes the network more the further off it is from the correct label.
Dropout to prevent overfitting, have some wiggle room,
"""


"""
Generalization of training example with step 3
"""
step = 3
inputs = []
outputs = []
for i in range(0, len(text) - max_len, step):
    inputs.append(text[i:i+max_len])
    outputs.append(text[i+max_len])


""" Map Each character to a label and create a reverse mapping to use later: """
char_labels = {ch:i for i, ch in enumerate(chars)}
labels_char = {i:ch for i, ch in enumerate(chars)}


""" Numerical input 3-tensor and output Matrix, Each sequence of characters is turned into a matrix of one-hot vectors """


# using bool to reduce memory usage
X = np.zeros((len(inputs), max_len, len(chars)), dtype=np.bool)
y = np.zeros((len(inputs), len(chars)), dtype=np.bool)

# set the appropriate indices to 1 in each one-hot vector
for i, example in enumerate(inputs):
    for t, char in enumerate(example):
        X[i, t, char_labels[char]] = 1
    y[i, char_labels[outputs[i]]] = 1


""" Function to produce text from the network"""

def generate(temperature=0.35, seed=None, predicate=lambda x: len(x) < 200):
    if seed is not None and len(seed) < max_len:
        raise Exception('Seed text must be at least {} chars long'.format(max_len))

    # if no seed text is specified, randomly select a chunk of text
    else:
        start_idx = random.randint(0, len(text) - max_len - 1)
        seed = text[start_idx:start_idx + max_len]

    sentence = seed
    generated = sentence

    while predicate(generated):
        # generate the input tensor
        # from the last max_len characters generated so far
        x = np.zeros((1, max_len, len(chars)))
        for t, char in enumerate(sentence):
            x[0, t, char_labels[char]] = 1.

        # this produces a probability distribution over characters
        probs = model.predict(x, verbose=0)[0]

        # sample the character to use based on the predicted probabilities
        next_idx = sample(probs, temperature)
        next_char = labels_char[next_idx]

        generated += next_char
        sentence = sentence[1:] + next_char
    return generated

""" Temperate controls the randomness of the network. The lower the temparature
    favors more likely values, higher introduce more randomness """

def sample(probs, temperature):
    """samples an index from a vector of probabilities
    (this is not the most efficient way but is more robust)"""
    a = np.log(probs)/temperature
    dist = np.exp(a)/np.sum(np.exp(a))
    choices = range(len(probs))
    return np.random.choice(choices, p=dist)


""" With this generation function we can modify how we train the newtork so that we see some output at each step:"""

epochs = 10
for i in range(epochs):
    print('epoch %d'%i)

    # set nb_epoch to 1 since we're iterating manually
    # comment this out if you just want to generate text
    model.fit(X, y, batch_size=128, nb_epoch=1)

    # preview
    for temp in [0.2, 0.5, 1., 1.2]:
        print('temperature: %0.2f'%temp)
        print('%s'%generate(temperature=temp))



# Add perplexity as evaluation measure. (A low perplexity indicates the probability distribution is good at predicting
# the sample.)

from keras import backend as K

def perplexity(y_true, y_pred, mask=None):
    if mask is not None:
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        mask = K.permute_dimensions(K.reshape(mask, y_true.shape[:-1]), (0, 1, 'x'))
        truth_mask = K.flatten(y_true*mask).nonzero()[0]  ### How do you do this on tensorflow?
        predictions = K.gather(y_pred.flatten(), truth_mask)
        return K.pow(2, K.mean(-K.log2(predictions)))
    else:
        return K.pow(2, K.mean(-K.log2(y_pred)))


    # from predicted probability distribution (ich glaube das isch "probs"), get the probability of the correct character
    # (and not the probability of the predicted character!!), das isch denn s'"y_pred" wo id endformle muss...
    # für was mer de rest brucht weiss ich nööööööd

    #TODO get probability of correct character from probs.... wie? y?

# http://stackoverflow.com/questions/37089201/how-to-calculate-perplexity-for-a-language-model-trained-using-keras