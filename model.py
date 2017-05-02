from keras.models import Model
from keras.layers import Input, Dense, Dropout, merge
from keras import optimizers
import keras.backend as K


def abs_diff(X):
    s = X[0]
    for i in range(1, len(X)):
        s -= X[i]
    s = K.abs(s)
    return s


def make_models(input_dim, embedding_size=32, embedding_bias=False,
                embedding_dropout=0):
    input_shape = (input_dim,)
    input_x = Input(shape=input_shape)
    embedding = Dense(embedding_size, use_bias=embedding_bias)(input_x)
    if embedding_dropout:
        embedding = Dropout(embedding_dropout)(embedding)
    embedding_model = Model(input_x, embedding)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    embedding_a = embedding_model(input_a)
    embedding_b = embedding_model(input_b)

    diff = merge([embedding_a, embedding_b], mode=abs_diff,
                 output_shape=(embedding_size,))
    output = Dense(1, activation='sigmoid')(diff)
    siamese_model = Model(input=[input_a, input_b], output=output)
    return embedding_model, siamese_model


if __name__ == "__main__":
    import numpy as np
    rng = np.random.RandomState(42)
    n_samples = 200
    n_features = int(1e5)
    a = rng.randn(n_samples, n_features)
    b = rng.randn(n_samples, n_features)

    embedding_size = 32
    embedding_model, siamese_model = make_models(
        n_features, embedding_size=embedding_size)

    # Smoke tests: check the output shape of predictions of randomly
    # initialized models
    assert embedding_model.predict(a).shape == (n_samples, embedding_size)
    assert siamese_model.predict([a, b]).shape == (n_samples, 1)

    # Check that we can fit balanced random labels
    y = rng.randint(low=0, high=2, size=n_samples)
    optimizer = optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)
    # optimizer = optimizers.Adam(lr=0.00001)
    siamese_model.compile(optimizer=optimizer, loss='binary_crossentropy',
                          metrics=['accuracy'])
    trace = siamese_model.fit([a, b], y, validation_split=0.2, epochs=30)

    # The model should be able to overfit the training data
    assert trace.history['loss'][-1] < 1e-4
    assert trace.history['acc'][-1] >= 0.95

    # The test accuracy should be random
    assert 0.35 <= trace.history['val_acc'][-1] <= 0.65
