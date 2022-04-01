# https://pywavelets.readthedocs.io/en/latest/ref/cwt.html
# print(list.shape) for sizing

from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential
from tensorflow import keras
from sklearn import metrics
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from numpy import genfromtxt
import pywt  # pip install PyWavelets
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import seaborn as sns
import timeit
timerStart = timeit.default_timer()


thesisData = genfromtxt('D:\\TensorFlow\\.venv\\thesisdata.csv', delimiter=',')
thesisY, thesisX = np.split(thesisData, 2, axis=1)
x_train, x_test, y_train, y_test = train_test_split(
    thesisX, thesisY, train_size=0.70, random_state=1)

mod_y_train = []
mod_y_test = []
LABEL_NAMES = ["Non-volatile", "Volatile"]

for x in range(0, y_train.size-y_train.size % 128, 128):
    total = 0
    for y in range(x, x+128, 1):
        total += abs(y_train[x])
    meanVolatility = total/128
    if meanVolatility < 0.3:
        label = 0
    else:
        label = 1
    mod_y_train.append(label)

complete_y_train = np.array(mod_y_train)

for x in range(0, y_test.size-y_test.size % 128, 128):
    total = 0
    for y in range(x, x+128, 1):
        total += abs(y_train[x])
    meanVolatility = total/128
    if meanVolatility < 0.3:
        label = 0
    else:
        label = 1
    mod_y_test.append(label)

complete_y_test = np.array(mod_y_test)


rescale_size = 64
n_scales = 64


def create_cwt_images(X, rescale_size):
    coeffs = []
    for x in range(0, X.size-X.size % 128, 128):
        x_sample = X[x:x+128, 0]
        coef, freqs = pywt.cwt(x_sample, np.arange(1, 65), 'morl')
        rescale_coeffs = resize(
            coef, (rescale_size, rescale_size), mode='constant')
        coeffs.append(rescale_coeffs)
        new_coeffs = np.array(coeffs)

    return new_coeffs


X_train_cwt = create_cwt_images(x_train, rescale_size)
print(
    f"shapes (n_samples, x_img, y_img) of X_train_cwt: {X_train_cwt.shape}")
X_test_cwt = create_cwt_images(x_test, rescale_size)
print(
    f"shapes (n_samples, x_img, y_img) of X_test_cwt: {X_test_cwt.shape}")

# Uncomment below to print plots
# plots = []
# for x in range(0, len(X_train_cwt), 1):
#     newplot = plt.matshow(X_train_cwt[x])
#     plots.append(newplot)
#     plt.show()

print(X_train_cwt.shape)

model = Sequential()


# 2 Convolution layer with Max polling
model.add(Conv2D(32, 5, activation="relu",
          padding='same', input_shape=(64, 64, 1)))
model.add(MaxPooling2D())
model.add(Conv2D(64, 5, activation="relu",
          padding='same', kernel_initializer="he_normal"))
model.add(MaxPooling2D())
model.add(Flatten())

# 3 Full connected layer
model.add(Dense(128, "relu", kernel_initializer="he_normal"))
model.add(Dense(54, "relu", kernel_initializer="he_normal"))
model.add(Dense(2, activation='softmax'))  # 6 classes

# summarize the model
print(model.summary())

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy']
)

callbacks = [ModelCheckpoint(
    filepath='best_model.h5', monitor='val_sparse_categorical_accuracy', save_best_only=True)]

print(complete_y_test)


history = model.fit(x=X_train_cwt,
                    y=complete_y_train,
                    batch_size=32,
                    epochs=15,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=(X_test_cwt, complete_y_test))

trained_cnn_model = model
cnn_history = history

y_pred_raw = trained_cnn_model.predict(X_test_cwt)
y_pred = np.where(y_pred_raw > 0.9, 1, 0)
y_pred = y_pred[:, 0]
confmat = metrics.confusion_matrix(y_true=complete_y_test, y_pred=y_pred)

fig, ax = plt.subplots(figsize=(3, 3))
ax.imshow(confmat, cmap=plt.cm.Blues, alpha=0.5)

n_labels = len(LABEL_NAMES)
ax.set_xticks(np.arange(n_labels))
ax.set_yticks(np.arange(n_labels))
ax.set_xticklabels(LABEL_NAMES)
ax.set_yticklabels(LABEL_NAMES)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=i, y=j, s=confmat[i, j], va='center', ha='center')

bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)

ax.set_title("Confusion Matrix")
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
plt.tight_layout()
plt.show()

print(y_pred_raw)
print(y_pred)
print(complete_y_test)

accuracy = metrics.accuracy_score(complete_y_test, y_pred)

print("Accuracy: %.2f%%" % (accuracy * 100.0))

timerStop = timeit.default_timer()
print("Running Time:", timerStop - timerStart)
