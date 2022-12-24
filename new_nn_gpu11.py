#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import IPython.display as display
from sklearn.model_selection import train_test_split
from keras.utils.vis_utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping


# In[32]:


if len(tf.config.list_physical_devices('GPU')) != 0:
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
    print(f"Device name: {tf.test.gpu_device_name()}")
else:
    print(f"GPU unavailable.")


# ### Датасет

# In[33]:


# data = pd.read_csv("./dataset.csv", delimiter=";")
data = pd.read_csv("./newNewAngle_swap.csv", delimiter=",")


# In[34]:


print(f"Shape: {data.shape} \n")
print(f"{data.head()}")


# In[35]:


print(f"{data.describe()}")


# In[36]:


feature_columns = data.columns[:12]
target_columns = data.columns[12:]

# feature_columns = data.columns[:3]
# target_columns = data.columns[3:]

print(f"Feature columns: {feature_columns}")
print(f"Target columns: {target_columns}")


# In[37]:


data.drop_duplicates()
print(f"Shape: {data.shape}")


# In[38]:


X = data.loc[:, feature_columns]
Y = data.loc[:, target_columns]

print(f"{X.head()}")
print(f"{Y.head()}")


# In[39]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, shuffle=1, random_state=42)
print(f"Types: {type(X_train)} \n")
print(f"X_train shape: {X_train.shape}")
print(f"Y_train shape: {Y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Y_test shape: {Y_test.shape}")


# In[53]:


print(X_train.values[0])
print('--------')
print(np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]))[0])


# In[14]:


X_train = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test.values, (X_test.shape[0], 1, X_test.shape[1]))

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")


# In[ ]:





# In[15]:


X_train_tensor = tf.convert_to_tensor(X_train)
Y_train_tensor = tf.convert_to_tensor(Y_train)
X_test_tensor = tf.convert_to_tensor(X_test)
Y_test_tensor = tf.convert_to_tensor(Y_test)
print(f"Types: {type(X_train_tensor)} \n")
print(f"{X_train_tensor}")


# In[16]:


# Слой нормализации преобразует входные данные с E=0 (мат.ожидание/expected value) и sd=1 (стандартное отклонение/standart deviation).
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(X_train)
print(f"Whether the normalization layer has been adapted: {normalizer.is_adapted}")


# In[ ]:


def my_loss_fn(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    print(y_true)
    print(y_pred)
    return tf.reduce_mean(squared_difference, axis=-1)


# In[57]:


def get_model(n_inputs, n_outputs, learning_rate, hidden_layer_count):
    """
    n_inputs - количество входов
    n_outputs - количество выходов
    learning_rate - скорость обучения
    hidden_layer_count - количество нейронов скрытого слоя
    """
    print(n_outputs)
    model = tf.keras.Sequential([
        normalizer,
        tf.keras.layers.LSTM(64, input_shape=(X_train_tensor.shape[1:]), activation='tanh', return_sequences=True),
        tf.keras.layers.LSTM(128, activation='tanh', return_sequences=True),
        tf.keras.layers.LSTM(256, activation='tanh', return_sequences=True),
        tf.keras.layers.LSTM(256, activation='tanh'),
        tf.keras.layers.Dense(32, activation='tanh'),
        tf.keras.layers.Dense(n_outputs, activation='linear')
    ])
    model.compile(
        optimizer=tf.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=False,
            name="Adam"
        ),
        loss=my_loss_fn,
        metrics=["accuracy"],
    )
    return model


# In[58]:


def train(learning_rate, epochs, hidden_layer_count, verbose=1):
    """
    learning_rate - скорость обучения
    epochs - количество итераций
    hidden_layer_count - количество нейронов скрытого слоя
    """
    # Build the model
    model = get_model(n_inputs=X_train.shape[1], n_outputs=Y_train.shape[1],
                      learning_rate=learning_rate, hidden_layer_count=hidden_layer_count)

    callbacks = [EarlyStopping(monitor='val_accuracy', patience = 5)]

    # Train the model
    history = model.fit(
        x=X_train_tensor,
        y=Y_train_tensor,
        callbacks = callbacks,
        epochs=epochs,
        verbose=verbose,
        validation_data=(X_test, Y_test)
    )

    loss = history.history["loss"][-1]
    accuracy = history.history["accuracy"][-1]

    if verbose == 0:
        print(f"loss: {loss} - accuracy: {accuracy}")

    return model, history, accuracy, loss


# In[30]:


print(X_train)


# In[ ]:


model, history, base_accuracy, base_loss = train(learning_rate=0.001, epochs=3, hidden_layer_count=32)


# In[136]:


print(f"Summary:\n{model.summary()}")
plot_model(model, show_shapes=True, show_layer_names=True)


# In[137]:


# Save the model to disk
# base_model.save_weights("model.h5")

# Load the model from disk later using:
# base_model.load_weights("model.h5")

# Evaluate the model
model.evaluate(
    X_test_tensor,
    Y_test_tensor
)

# Predict on the first 5 tests
predictions = model.predict(X_test[:5])

# Print our model's predictions
print(predictions)

# Check our predictions against the ground truths
print(Y_test[:5])


# In[138]:


Y_pred = model.predict(X_test)

training_loss = history.history["loss"]
test_loss = history.history["val_loss"]

# Get training and test accuracy histories
training_acc = history.history["accuracy"]
test_acc = history.history["val_accuracy"]

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)


# In[139]:


# Visualize loss history
plt.figure()
plt.title("Loss")
plt.plot(epoch_count, training_loss)
plt.plot(epoch_count, test_loss)
plt.legend(["Train", "Test"])
plt.xlabel("Epoch")
plt.ylabel("Loss value")
plt.show()

# Visualize accuracy history
plt.figure()
plt.title("Acuracy")
plt.plot(epoch_count, training_acc)
plt.plot(epoch_count, test_acc)
plt.legend(["Train", "Test"])
plt.xlabel("Epoch")
plt.ylabel("Accuracy value")
plt.show()


# In[144]:


model.save('RNN_model_1.tf', save_format="tf")


# In[145]:


model_test = tf.keras.models.load_model('RNN_model_1.tf')
type(model_test)

