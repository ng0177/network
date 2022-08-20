import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from numpy import sqrt
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split



# reading data -----------------------------------------------------------


# train data -------------------------------------------------------------

train = pd.read_csv("data_set.csv")
train = train.values

train_x, train_y = train[:, :-2], train[:, -2:]


print("x shape is ", train_x.shape)
print("y shape is ", train_y.shape)

a,b = train_y.shape



# validation data --------------------------------------------------------

validation = pd.read_csv("data_val.csv")
validation = validation.values

validation_x, validation_y = validation[:, :-2], validation[:, -2:]




# test data --------------------------------------------------------------

# Data:   pt, Tt, dpdt, tau, beta
# Input:  pt, Tt, dpdt
# Output: tau, beta

test = pd.read_csv("data_test.csv")
test = test.values

test_x, test_y = test[:, :-2], test[:, -2:]




# Normalization -------------------------------------------------------
normalization = 1

if normalization == 1:

	data_set = np.concatenate((train,validation), axis=0)
	data_set = np.concatenate((data_set,test), axis=0) 

	mean_train = np.mean(train, axis=0) 
	std_train  = np.std(train, axis=0)
	
	train_x    = (train_x - mean_train[:-b]) / std_train[:-b]
	train_y    = (train_y - mean_train[-b:]) / std_train[-b:]
	
	validation_x = (validation_x - mean_train[:-b]) / std_train[:-b]
	validation_y = (validation_y - mean_train[-b:]) / std_train[-b:]
	
	test_x = (test_x - mean_train[:-b]) / std_train[:-b]
   
# -----------------------------------------------------------------------
# Neural Network Model  -------------------------------------------------

###
n_features = train_x.shape[1]
inputs  = keras.Input(shape=(n_features,))
dense   = layers.Dense(64, activation="relu")
x       = dense(inputs)
x       = layers.Dense(64, activation="relu")(x)
x       = layers.Dense(16, activation="relu")(x)
outputs = layers.Dense(2)(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='mse')
model.summary()
###

# fit the model
history = model.fit(train_x, train_y, batch_size=64, epochs=5, validation_data=(validation_x, validation_y))

# evaluate the model
error = model.evaluate(validation_x, validation_y, verbose=0)
print('MSE: %.5f, RMSE: %.5f' % (error, sqrt(error)))

res = model.predict(test_x)

# Reconstruct output variables
if normalization == 1:
 
	res1 = res[:,0] * std_train[-b]   + mean_train[-b]
	res2 = res[:,1] * std_train[-b+1] + mean_train[-b+1]


# Plot Results  ---------------------------------------------------------


plt.figure(1)
plt.plot(res1, label="NN")
plt.plot(test_y[:,0], label="CFD")
plt.xlabel(r"$t / t_c$")
plt.ylabel(r"$\tau$")
plt.legend()
plt.savefig('torque.pdf')


plt.figure(2)
plt.plot(res2, label="NN")
plt.plot(test_y[:,1], label="CFD")
plt.xlabel(r"$t / t_c$")
plt.ylabel(r"$\beta$")
plt.legend()
plt.savefig('beta.pdf')


plt.figure(3)
plt.plot(np.sqrt(history.history['loss']))
plt.plot(np.sqrt(history.history['val_loss']))
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.xlim((0,100))
plt.legend(['train','val'], loc = 'upper left')
plt.savefig('error_long.pdf')



#df = pd.DataFrame(history.history['loss'])
#df.to_csv('training_loss.csv',header=["training_loss"],index=False)

#df = pd.DataFrame(history.history['val_loss'])
#df.to_csv('validation_loss.csv',header=["validation_loss"],index=False)







