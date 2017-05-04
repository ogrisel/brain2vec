from matplotlib import pyplot as plt
from nilearn import plotting
from nilearn.input_data import NiftiMasker
from nilearn.image import index_img
from data import generate_learning_set
import os
import numpy as np
import nibabel


data_root_path = './hcp_olivier'
subject_id = '102816'

# Localize data through file system relative indexing method
path = os.path.join(data_root_path, subject_id, 'MNINonLinear', 'Results', 'rfMRI_REST1_LR', 'rfMRI_REST1_LR.npy')

# Create a masker to unmask data
masker = NiftiMasker(mask_img=os.path.join(data_root_path, 'mask_img.nii.gz'))
masker.fit()

atlas = masker.inverse_transform(np.load(path)[:32])

# Plot 32 brains
for i in range(32):
    plotting.plot_stat_map(index_img(atlas, i), output_file='brain_{}.png'.format(i))