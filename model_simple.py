from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Merge

import we_helpers as wh


def get_model():
    model = Sequential()

    model.add(Conv2D(8, (3, wh.WORD_EMBEDDING_SIZE), activation='relu', input_shape=(wh.MAX_SENTENCE_LENGTH, wh.WORD_EMBEDDING_SIZE, 1)))
    model.add(MaxPooling2D(pool_size=(58, 1)))

    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model
