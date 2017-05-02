from keras.models import Model
from keras.layers import Input, Dense, Dropout, merge
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
    n_samples = 100
    n_features = int(1e5)
    a = rng.randn(n_samples, n_features)
    b = rng.randn(n_samples, n_features)
    y = rng.randint(low=0, high=2, size=n_samples)

    embedding_model, siamese_model = make_models(n_features)
    print(embedding_model.predict(a).shape)
    print(siamese_model.predict([a, b]).shape)
