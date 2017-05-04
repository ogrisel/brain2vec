from nilearn import plotting
from nilearn.input_data import NiftiMasker
from data import generate_learning_set
import os
import numpy as np


data_root_path = './hcp_olivier'
subject_id = '102816'

# Localize data through file system relative indexing method
path = os.path.join(data_root_path, subject_id, 'MNINonLinear', 'Results', 'rfMRI_REST1_LR', 'rfMRI_REST1_LR.npy')

# Create a masker to unmask data
masker = NiftiMasker(mask_img=os.path.join(data_root_path, 'mask_img.nii.gz'))
masker.fit()

# Take the diffs
a, b, _ = generate_learning_set(np.load(path), random_permutation=False, offset=0)
diffs = b - a

# Plot the diffs
plotting.plot_stat_map(masker.inverse_transform(diffs[0]))
# plotting.show()

print(diffs.shape)
masker.inverse_transform(diffs[:100]).to_filename('diffs.nii.gz')