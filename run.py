import data
import model
import numpy as np
from keras import optimizers
from sklearn.decomposition import PCA
from joblib import Memory


data_root_path = './hcp_olivier'
subject_id = '102816'

# Localize data through file system relative indexing method
path = os.path.join(data_root_path, subject_id, 'MNINonLinear', 'Results', 'rfMRI_REST1_LR', 'rfMRI_REST1_LR.npy')

m = Memory(cachedir='/tmp')


@m.cache
def pca_transform(x, n_components=128, seed=0):
    return PCA(n_components=n_components, svd_solver='randomized',
               random_state=seed).fit_transform(x)


# Use data loading library to load data
# x = np.load(path)
# print("extracting pca components:")
# x = pca_transform(x)
# a, b, y = data.generate_learning_set(x)


def make_synthetic_data(n_samples, n_signal_features, n_noise_features,
                        seed=42):
    rng = np.random.RandomState(seed)
    signal_features = np.linspace(-0.1, 0.1, n_samples).reshape(-1, 1)
    signal_features = signal_features * rng.uniform(0.5, 1.5, n_signal_features)
    noise_features = rng.randn(n_samples, n_noise_features)
    return np.hstack([signal_features, noise_features])


x = make_synthetic_data(1000, 100, 300)
a, b, y = data.generate_learning_set(x)

# Generate the model
embedding_model, siamese_model = model.make_linear_models(a.shape[1])

# optimizer = optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)
optimizer = optimizers.Adam(lr=0.0001)

siamese_model.compile(optimizer=optimizer, loss='binary_crossentropy',
                      metrics=['accuracy'])

print("data shapes:")
print(a.shape)
print(b.shape)
print(y.shape)

trace = siamese_model.fit([a, b], y, validation_split=0.2, epochs=300,
                          batch_size=32, shuffle=True)

print(trace.history['acc'][-1])
print(trace.history['val_acc'][-1])
