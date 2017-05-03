import data
import model
import numpy as np
from keras import optimizers


data_root_path = './hcp_olivier'
subject_id = '102816'

# Localize data through file system relative indexing method
path = os.path.join(data_root_path, subject_id, 'MNINonLinear', 'Results', 'rfMRI_REST1_LR', 'rfMRI_REST1_LR.npy')

# Use data loading library to load data
a, b, y = data.generate_learning_set(np.load(path))

# Generate the model
embedding_model, siamese_model = model.make_linear_models(a.shape[1])

optimizer = optimizers.SGD(lr=0.00001, momentum=0.9, nesterov=True)
# optimizer = optimizers.Adam(lr=0.0001)

siamese_model.compile(optimizer=optimizer, loss='binary_crossentropy',
                      metrics=['accuracy'])

print("data shapes:")
print(a.shape)
print(b.shape)
print(y.shape)

trace = siamese_model.fit([a, b], y, validation_split=0.2, epochs=30,
                          batch_size=16, shuffle=True)

print(trace.history['acc'][-1])
print(trace.history['val_acc'][-1])
