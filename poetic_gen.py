import random
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
from keras.optimizers import RMSprop


###################################################################
##                  get a training dataset                       ##
###################################################################
# get a text file directly from tensorflow
filepath = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
# get the text from the file in the script
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()

###################################################################
## convert the text into numerical format for the neural network ##
###################################################################
text = text[300000:800000]  # truncate what we are using for training
characters = sorted(set(text))  # filter all unique characters in the set
# create a dictionary which has the character as a key, index as a value for all the indices and
# the characters in the enumeration of characters. The enumeration assigns a number to each character in the set.
char_to_index = dict((c, i) for i, c in enumerate(characters))
# reverse of above
index_to_char = dict((i, c) for i, c in enumerate(characters))
# next characters are the target and the sentences are the features. Load a sentence in and the result should be
# the next character. "How are yo" -> 'u'.
SEQ_LENGTH = 40
STEP_SIZE = 3
'''
sentences = []  # empty list of sentences
next_characters = []  # empty list of next characters
# we are getting training examples with sentences and the next correct letter. For example, if SEQ_LENGTH is 5 then we
# get characters 0 through 4 and then the character at 5 is the next character.
for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
    sentences.append(text[i: i+SEQ_LENGTH])
    next_characters.append(text[i+SEQ_LENGTH])
# a numpy array full of zeros that is the number of sentences x length of the sentences x number of characters.
# one dimension for all the possible sentences, one for all the positions in these sentences, and one for each possible
# character. In each sentence, when a certain character occurs it is set to true and all other values remain 0.
x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=bool)
# which is the next character for the next sentence?
y = np.zeros((len(sentences), len(characters)), dtype=bool)
for i, sentence in enumerate(sentences):
    for t, character in enumerate(sentence):
        x[i, t, char_to_index[character]] = 1  # for sentence #i, position #t, and character #index set true/1
    y[i, char_to_index[next_characters[i]]] = 1  # for sentence #i, the correct next character is character #index
'''
###################################################################
##             Build the neural network model                    ##
###################################################################
'''
# Train the model
model = Sequential()
model.add(LSTM(128, input_shape=(SEQ_LENGTH, len(characters))))  # Long Short Term Memory, LSTM(# of neurons, data_shape)
model.add(Dense(len(characters)))  # has as many neurons as we have possible characters
# softmax scales the output so that all the values add up to 1 and gives the likelihood of each character as the next character
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.01))
model.fit(x, y, batch_size=256, epochs=4)
model.save('poeticgen.model')
'''
model = tf.keras.models.load_model('poeticgen.model')


# From keras tutorial helper function
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds/np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# if you want a text that is completely generated, you have to cut off the first 40 characters
def generate_text(length, temperature):
    start_index = random.randint(0, len(text) - SEQ_LENGTH - 1)
    generated = ''
    sentence = text[start_index: start_index + SEQ_LENGTH]
    generated += sentence
    for i in range(length):
        x = np.zeros((1, SEQ_LENGTH, len(characters)))
        for t, character in enumerate(sentence):
            x[0, t, char_to_index[character]] = 1
        predictions = model.predict(x, verbose=0)[0]
        next_index = sample(predictions, temperature)
        next_character = index_to_char[next_index]
        generated += next_character
        sentence = sentence[1:] + next_character
    return generated


print('-------------0.2----------------')
print(generate_text(300, 0.2))
print('-------------0.4----------------')
print(generate_text(300, 0.4))
print('-------------0.8----------------')
print(generate_text(300, 0.8))
print('-------------1.0----------------')
print(generate_text(300, 1.0))
print('-------------1.2----------------')
print(generate_text(300, 1.2))
print('-------------1.4----------------')
print(generate_text(300, 1.4))
print('-------------1.6----------------')
print(generate_text(300, 1.6))
print('-------------1.8----------------')
print(generate_text(300, 1.8))

