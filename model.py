from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, merge
from keras import optimizers


def merge_diff(X):
    s = X[0]
    for i in range(1, len(X)):
        s -= X[i]
    return s


def merge_mul(X):
    m = X[0]
    for i in range(1, len(X)):
        m *= X[i]
    return m


def make_linear_models(input_dim, embedding_size=32, embedding_bias=False,
                       embedding_dropout=0):
    """Build a linear Siamese model on abs difference of embeddings.

    The output is a sigmoid activation (binary logistic regression) to predict
    forward order or bacward order relationships between input a and b.
    """
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

    diff = merge([embedding_a, embedding_b], mode=merge_diff,
                 output_shape=(embedding_size,))
    diff = BatchNormalization()(diff)
    output = Dense(1, activation='sigmoid')(diff)
    siamese_model = Model([input_a, input_b], output)
    return embedding_model, siamese_model


def make_mlp_models(input_dim, embedding_size=32, embedding_bias=False,
                    embedding_dropout=0.2, hidden_size=128,
                    n_hidden=2, dropout=0.2):
    """Non-linear Siamese model on a pair of embeddings

    Compares a pair of embeddings using a concatenation of the 2 embeddings,
    their elementwise absolute difference and multiplicative
    interactions.BaseException

    The classifier a 2 hidden layers feed forward neural network with relu
    activations.

    The output is a sigmoid activation (binary logistic regression) to predict
    forward order or bacward order relationships between input a and b.
    """
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

    diff = merge([embedding_a, embedding_b], mode=merge_diff,
                 output_shape=(embedding_size,))
    mul = merge([embedding_a, embedding_b], mode=merge_mul,
                output_shape=(embedding_size,))
    x = merge([embedding_a, embedding_b, diff, mul], mode='concat')
    x = BatchNormalization()(x)
    for i in range(n_hidden):
        x = Dense(hidden_size, activation='relu')(x)
        x = Dropout(dropout)(x)
    output = Dense(1, activation='sigmoid')(x)
    siamese_model = Model([input_a, input_b], output)
    return embedding_model, siamese_model


def make_permutation_models(input_dim, n_inputs=2, embedding_size=32,
                            embedding_bias=False, embedding_dropout=0.2,
                            hidden_size=128, n_hidden=2, dropout=0.2):
    """Take n frames as input and predict if they are in the correct order"""
    input_shape = (input_dim,)
    input_x = Input(shape=input_shape)
    embedding = Dense(embedding_size, use_bias=embedding_bias)(input_x)
    if embedding_dropout:
        embedding = Dropout(embedding_dropout)(embedding)
    embedding_model = Model(input_x, embedding)

    inputs = []
    embeddings = []
    for i in range(n_inputs):
        input_i = Input(shape=input_shape)
        inputs.append(input_i)
        embedding_i = embedding_model(input_i)
        embeddings.append(embedding_i)
    x = merge(embeddings, mode='concat')
    x = BatchNormalization()(x)
    for i in range(n_hidden):
        x = Dense(hidden_size, activation='relu')(x)
        x = Dropout(dropout)(x)
    output = Dense(1, activation='sigmoid')(x)
    siamese_model = Model(inputs, output)
    return embedding_model, siamese_model


if __name__ == "__main__":
    import numpy as np
    rng = np.random.RandomState(42)
    n_samples = 200
    n_features = int(1e5)
    a = rng.randn(n_samples, n_features)
    b = rng.randn(n_samples, n_features)

    embedding_size = 32

    for model_factory in [make_linear_models, make_mlp_models,
                          make_permutation_models]:
        embedding_model, siamese_model = model_factory(
            n_features, embedding_size=embedding_size)

        # Smoke tests: check the output shape of predictions of randomly
        # initialized models
        assert embedding_model.predict(a).shape == (n_samples, embedding_size)
        assert siamese_model.predict([a, b]).shape == (n_samples, 1)

        # Check that we can fit balanced random labels
        y = rng.randint(low=0, high=2, size=n_samples)
        optimizer = optimizers.SGD(lr=0.0001, momentum=0.9, nesterov=True)
        # optimizer = optimizers.Adam(lr=0.0001)
        siamese_model.compile(optimizer=optimizer, loss='binary_crossentropy',
                              metrics=['accuracy'])
        trace = siamese_model.fit([a, b], y, validation_split=0.2, epochs=30,
                                  batch_size=16)

        # The model should be able to overfit the training data
        assert trace.history['acc'][-1] >= 0.95

        # The test accuracy should be random because the labels are random
        assert 0.3 <= trace.history['val_acc'][-1] <= 0.7
