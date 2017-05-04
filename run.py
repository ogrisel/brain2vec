import os.path as op
import data
import model
import numpy as np
from keras import optimizers
from sklearn.decomposition import PCA
from joblib import Memory


data_root_path = './hcp_olivier'
# subject_ids = ['102816', '102311']
subject_ids = ['102816']
session_ids = [1, 2]
# session_ids = [1]


def get_paths(data_root_path, subject_ids, session_ids=[1, 2]):
    all_paths = []
    for subject_id in subject_ids:
        for session_id in session_ids:
            path = op.join(data_root_path, subject_id,
                           'MNINonLinear', 'Results',
                           'rfMRI_REST%d_LR' % session_id,
                           'rfMRI_REST%d_LR.npy' % session_id)
            all_paths.append(path)
    return all_paths


m = Memory(cachedir='/tmp')


@m.cache
def pca_transform(x, n_components=128, seed=0):
    return PCA(n_components=n_components, svd_solver='randomized',
               random_state=seed).fit_transform(x)


# Use data loading library to load data
all_paths = get_paths(data_root_path, subject_ids, session_ids=session_ids)
x = np.vstack([np.load(path).astype(np.float32) for path in all_paths])
print("original data shape:")
print(x.shape)
# print("extracting pca components:")
# x = pca_transform(x)
# a, b, y = data.generate_learning_set(x, scans_to_average=2)


def make_synthetic_data(n_samples, n_signal_features, n_noise_features,
                        seed=42):
    rng = np.random.RandomState(seed)
    signal_features = np.linspace(-0.1, 0.1, n_samples).reshape(-1, 1)
    signal_features = signal_features * rng.uniform(0.5, 1.5, n_signal_features)
    noise_features = rng.randn(n_samples, n_noise_features)
    return np.hstack([signal_features, noise_features])


# x = make_synthetic_data(1000, 100, 0)
# a, b, y = data.generate_learning_set(x)
# input_dim = a.shape[1]
window_size = 5
scans, labels = data.generate_permutations(x, scans_to_permute=window_size,
                                           backward_is_ok=True)
del x
input_dim = len(scans[0][0])

# Generate the model
embedding_model, siamese_model = model.make_permutation_models(
    input_dim, n_inputs=window_size)

# optimizer = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
optimizer = optimizers.Adam(lr=0.001)

siamese_model.compile(optimizer=optimizer, loss='binary_crossentropy',
                      metrics=['accuracy'])


trace = siamese_model.fit(scans, labels, validation_split=0.2, epochs=300,
                          batch_size=32, shuffle=True)

print(trace.history['acc'][-1])
print(trace.history['val_acc'][-1])
