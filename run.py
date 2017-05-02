import data
import model
import numpy as np
from keras import optimizers


# Localize data through file system relative indexing method
path = 'hcp_olivier/102816/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR.npy'

# Use data loading library to load data
a, b, y = data.generate_learning_set(np.load(path))

# Generate the model
embedding_model, siamese_model = model.make_mlp_models(a.shape[1], embedding_dropout=0.2)

optimizer = optimizers.SGD(lr=0.00001, momentum=0.9, nesterov=True)
# optimizer = optimizers.Adam(lr=0.0001)

siamese_model.compile(optimizer=optimizer, loss='binary_crossentropy',
                      metrics=['accuracy'])

print(a.shape)
print(a[:10])

trace = siamese_model.fit([a, b], y, validation_split=0.2, epochs=30,
                          batch_size=16)

print(trace.history['acc'][-1])
print(trace.history['val_acc'][-1])