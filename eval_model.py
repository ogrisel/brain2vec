import sys
import os.path as op
import data
import numpy as np
from keras.models import load_model


data_root_path = './hcp_olivier'
subject_ids = ['103414']
session_ids = [1]


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


all_paths = get_paths(data_root_path, subject_ids, session_ids=session_ids)
x = np.vstack([np.load(path).astype(np.float32) for path in all_paths])
print("original data shape:")
print(x.shape)
window_size = 5
scans, labels = data.generate_permutations(x, scans_to_permute=window_size,
                                           backward_is_ok=True)
del x
input_dim = len(scans[0][0])

print('loading model:')
model = load_model(sys.argv[1])
print('computing predictions on new data')
predicted = model.predict(scans)
print("accuracy: %0.3f" % np.mean((predicted > 0.5) == labels))
