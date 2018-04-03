import codecs
from bs4 import BeautifulSoup
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random, sys

# read file
fp = codecs.open("./BEXX0003.txt", mode ="r", encoding="utf-16")
soup = BeautifulSoup(fp, "html.parser")
body = soup.select_one("text")
text = body.getText() + " "
print('The length of corpus: ', len(text))

# read word one by one and give an ID.
chars = sorted(list(set(text)))
print(' number of using chars :', len(chars))
char_indices = dict((c,i) for i, c in enumerate(chars)) # char -> ID
indices_char = dict((i, c) for i, c in enumerate(chars)) # ID -> char

#
maxlen = 20
step = 3
sentences = []
next_chars = []

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i:i+maxlen])
    next_chars.append(text[i+maxlen])
print(' number of sentences to learning : ', len(sentences))
print(' conver text to ID vector ...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# build the model
print (' building the model... ')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))
optimizer = RMSprop(lr = 0.01)
model.compile(loss ='categorical_crossentropy', optimizer = optimizer)

# from array
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
# learinig and generate the text
for iteration in range(1, 60):
    print()
    print(' repeat =', iteration)
    model.fit(X, y, batch_size = 128, nb_epoch = 1)
    # random start text
    start_index = random.randint(0, len(text) - maxlen -1)
    #
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('--diversity =', diversity)
        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('-- seed = "'+sentence+ '""')
        sys.stdout.write(generated)

        # seed based text gen
        for i in range(400):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.
            # predict the next char
            preds = model.predict(x, verbose = 0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]
            # print
            generated += next_char
            sentence = sentence[1:] + next_char
            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

