import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

dataset11 = pd.read_csv("C:\\Users\\Admin\\Desktop\\Кандидат\\neural_network\\dataset.csv", delimiter=";")
dups = dataset11.duplicated(subset=['X','Y','Z'])
dataset2 = dataset11[~dups]
dataset = dataset2.values
dataset = (dataset - dataset.mean(axis=0, keepdims=True)) / (dataset.std(axis=0, keepdims=True))
#dataset = np.loadtxt("C:\\Users\\Admin\\source\\repos\\STL-Viewer\\RobotOmgtu\\yours.csv", delimiter=";", encoding='utf-8', dtype=None)

angles = 6

X_super = dataset[:20000,:3]
Y_super = dataset[:20000,3:]

X_super_test = dataset[90000:,:3]
Y_super_test = dataset[90000:,3:]
sss = X_super_test[0:1, :3]
model = Sequential()

# from sklearn.model_selection import train_test_split
#
# X_train, X_test, y_train, y_test = train_test_split(X_super, Y_super, test_size=0.15, shuffle=1, random_state=0)
#
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler(feature_range=(-1, 1))
# X_train = scaler.fit_transform(X_train)
# scaler = MinMaxScaler(feature_range=(-1, 1))
# X_test = scaler.fit_transform(X_test)
#
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.18, random_state=0)#0.25

model.add(Dense(3, activation='relu', input_shape=(3, )))
model.add(Dense(100, activation='relu'))
model.add(Dense(6, activation='linear'))

model.compile(optimizer='Adam', loss='mae')

model.summary()
#early_stop = EarlyStopping(monitor='accuracy', patience=15)
train_model=model.fit(
    x=X_super,
    y=Y_super,
    epochs=200,
    verbose=2,
    validation_data=(X_super_test, Y_super_test),
    callbacks=[tf.keras.callbacks.TensorBoard('logs/1/train')]
)

y_pred = model.predict(X_super_test[0:1, :3])

training_loss = train_model.history['loss']
test_loss = train_model.history['val_loss']

# Get training and test accuracy histories
training_acc = train_model.history['accuracy']
test_acc = train_model.history['val_accuracy']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
plt.figure()
plt.title('Loss')
plt.plot(epoch_count, training_loss)
plt.plot(epoch_count, test_loss)
plt.legend(['Train', 'Test'])
plt.xlabel('Epoch')
plt.ylabel('Loss value')
plt.show()

# Visualize accuracy history
plt.figure()
plt.title('Acuracy')
plt.plot(epoch_count, training_acc)
plt.plot(epoch_count, test_acc)
plt.legend(['Train', 'Test'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy value')
plt.show()


#model = load_model(r'C:\\Users\\Admin\\source\\repos\\STL-Viewer\\RobotOmgtu\\model.h5')
model.save(r'C:\\Users\\Admin\\source\\repos\\STL-Viewer\\RobotOmgtu\\model.h5')

aa = 'dasdasd'