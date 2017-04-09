from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Merge

import we_helpers as wh


def get_model():
    model = Sequential()

    model1 = Sequential()
    model1.add(Conv2D(8, (2, wh.WORD_EMBEDDING_SIZE), activation='relu', input_shape=(wh.MAX_SENTENCE_LENGTH, wh.WORD_EMBEDDING_SIZE, 1)))
    model1.add(MaxPooling2D(pool_size=(59, 1)))

    model2 = Sequential()
    model2.add(Conv2D(8, (3, wh.WORD_EMBEDDING_SIZE), activation='relu', input_shape=(wh.MAX_SENTENCE_LENGTH, wh.WORD_EMBEDDING_SIZE, 1)))
    model2.add(MaxPooling2D(pool_size=(58, 1)))

    model3 = Sequential()
    model3.add(Conv2D(8, (4, wh.WORD_EMBEDDING_SIZE), activation='relu', input_shape=(wh.MAX_SENTENCE_LENGTH, wh.WORD_EMBEDDING_SIZE, 1)))
    model3.add(MaxPooling2D(pool_size=(57, 1)))

    model4 = Sequential()
    model4.add(Conv2D(8, (5, wh.WORD_EMBEDDING_SIZE), activation='relu', input_shape=(wh.MAX_SENTENCE_LENGTH, wh.WORD_EMBEDDING_SIZE, 1)))
    model4.add(MaxPooling2D(pool_size=(56, 1)))

    model.add(Merge([model1, model2, model3, model4], mode='concat'))
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model
