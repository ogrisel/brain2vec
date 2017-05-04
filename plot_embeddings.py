from matplotlib import pyplot as plt
from nilearn import plotting
from nilearn.input_data import NiftiMasker
from nilearn.image import index_img
from data import generate_learning_set
import os
import os.path as op
import numpy as np
import nibabel
from keras.models import load_model


data_root_path = './hcp_olivier'
subject_id = '102816'
folder = 'visualizations'

weights = load_model('checkpoints/model.000-0.51.h5').get_weights()[0]
images = weights.T

# Localize data through file system relative indexing method
# path = os.path.join(data_root_path, subject_id, 'MNINonLinear', 'Results', 'rfMRI_REST1_LR', 'rfMRI_REST1_LR.npy')
# images = np.load(path)[:32]

# Create a masker to unmask data
masker = NiftiMasker(mask_img=os.path.join(data_root_path, 'mask_img.nii.gz'))
masker.fit()

atlas = masker.inverse_transform(images)

if not op.exists(folder):
    os.makedirs(folder)
# Plot 32 brains
for i in range(32):
    plotting.plot_stat_map(index_img(atlas, i),
                           output_file=folder + '/brain_{:02d}.png'.format(i))
