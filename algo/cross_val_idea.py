## import necessary libraries to run tensorflow models
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import kerastuner as kt
import keras.backend as kb

from keras.utils import to_categorical, plot_model
from keras.models import Sequential, Input, Model, load_model

from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv3D, MaxPooling3D, MaxPool3D, AvgPool3D, GlobalAveragePooling3D
from keras.layers import Input, LSTM, LSTMCell, TimeDistributed, Reshape, Concatenate, GRU, concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

from keras.optimizers import Adam
from keras.metrics import AUC, BinaryAccuracy, Precision
from keras.wrappers.scikit_learn import KerasClassifier

from kerastuner.tuners import RandomSearch, BayesianOptimization, Sklearn, Hyperband
from kerastuner.engine.hyperparameters import HyperParameters

from tensorflow_addons.metrics import CohenKappa

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV, TimeSeriesSplit
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer


## load tensor and labels
tensor = np.load('/Users/mticli/Documents/BOCCONI/FINAL PROJECT/CNN_LSTM-stock-prices-prediction/data_test/tensor.npy')
labels = np.load('/Users/mticli/Documents/BOCCONI/FINAL PROJECT/CNN_LSTM-stock-prices-prediction/data_test/labels.npy')


## define periods of training and out of sample data
fixed_starting_point = 50       ## ==> this cannot be changed and it must be 50 due to data preparation procedure
start_training = 4000
end_training = 4400
start_out_of_sample = 4400
end_out_of_sample = 4550


## training and test data
x_train, x_test, y_train, y_test = train_test_split(tensor[start_training:end_training, :, :, :], labels[start_training:end_training], train_size=0.75, random_state = 42, shuffle=False)
x_train, x_test = np.reshape(x_train, x_train.shape + (1,)), np.reshape(x_test, x_test.shape + (1,))
y_train, y_test = to_categorical(y_train), to_categorical(y_test)
input_shape = x_train.shape[1:]


## out of sample data
features_test, labels_test = tensor[start_out_of_sample:, :, :, :], labels[start_out_of_sample:]
labels_test = to_categorical(labels_test)


## nasdaq returns to be compared with
df = pd.read_csv('/Users/mticli/Documents/BOCCONI/FINAL PROJECT/CNN_LSTM-stock-prices-prediction/data_test/NASDAQ.csv')
df = df[fixed_starting_point:].reset_index()

df_out_of_sample = df[start_out_of_sample:].reset_index()


## class created to perform cross validation using Blocking Time Series method, different from Time Series Split of Scikit-learn
class BlockingTimeSeriesSplit():
	def __init__(self, n_splits):
		self.n_splits = n_splits

	def get_n_splits(self, X, y, groups):
		return self.n_splits

	def split(self, X, y=None, groups=None):
		n_samples = len(X)
		k_fold_size = n_samples // self.n_splits
		indices = np.arange(n_samples)

		margin = 0
		for i in range(self.n_splits):
			start = i * k_fold_size
			stop = start + k_fold_size
			mid = int(0.8 * (stop - start)) + start
			yield indices[start: mid], indices[mid + margin: stop]


## start cross validation of the model
def build_model(hp):
	inp = Input(shape=input_shape)
	x = inp
	y = inp

	for i in range(hp.Int('1_parallel_conv_blocks', 1, 2, default=1)):
		filters_1 = hp.Int('1_filters_' + str(i), 2, 20, step=2)
		x = Conv3D(filters_1, kernel_size=(hp.Int('1_1_kernel_'+str(i),1, 10), hp.Int('1_2_kernel_'+str(i),1, 10), hp.Int('1_3_kernel_'+str(i),1, 10)), padding='same', data_format='channels_last', activation='relu')(x)

	for j in range(hp.Int('2_parallel_conv_blocks', 1, 2, default=1)):
		filters_2 = hp.Int('2_filters_' + str(j), 2, 20, step=2)
		y = Conv3D(filters_2, kernel_size=(hp.Int('2_1_kernel_'+str(i),1, 10), hp.Int('2_2_kernel_'+str(i),1, 10), hp.Int('2_3_kernel_'+str(i),1, 10)), padding='same',  data_format='channels_last', activation='relu')(y)

	model = Concatenate(axis=-1)([x, y])

	for k in range(hp.Int('seq_conv_blocks', 1, 3, default = 2)):
		filters_3 = hp.Int('3_filters_' + str(k), 2, 20, step=2)
		model = Conv3D(filters_3, kernel_size=(hp.Int('3_1_kernel_'+str(i),1, 10), hp.Int('3_2_kernel_'+str(i),1, 10), hp.Int('3_3_kernel_'+str(i),1, 10)), padding='same', data_format='channels_last', activation='relu')(model)

	for pool_3 in range(hp.Int('3_pooling', 0, 1, default=0)):
		model = BatchNormalization()(model)
		if hp.Choice("3_pooling_" + str(pool_3), ["avg", "max"]) == "max":
			model = MaxPool3D()(model)
		else:
			model = AvgPool3D()(model)

	model = Flatten()(model)

	for dense_layers in range(hp.Int('layers', 0, 5)):
		model = Dense(units=hp.Int('units_' + str(dense_layers), 0, 5000, step=10), activation='relu')(model)

	out = Dense(2, activation=hp.Choice('final_activation', ['sigmoid', 'softmax']))(model)

	MODEL = Model(inputs=inp, outputs=out)

	MODEL.compile(optimizer=Adam(hp.Float("learning_rate", 1e-5, 1e-1, sampling="log")), loss='categorical_crossentropy', metrics=['accuracy', AUC()])

	return MODEL
  
class CVTuner(kt.engine.tuner.Tuner):
	def run_trial(self, trial, x, y, *args, **kwargs):
		cv = BlockingTimeSeriesSplit(n_splits=5)
		val_accuracy_list = []
		# batch_size = 32
		# epochs = 24
		batch_size = trial.hyperparameters.Int('batch_size', 2, 64, step=8)
		epochs = trial.hyperparameters.Int('epochs', 10, 100, step=10)

		for train_indices, test_indices in cv.split(x):
			x_train, x_test = x[train_indices], x[test_indices]
			y_train, y_test = y[train_indices], y[test_indices]
			model = self.hypermodel.build(trial.hyperparameters)
			model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
			val_loss, val_accuracy, val_auc = model.evaluate(x_test, y_test)
			val_accuracy_list.append(val_accuracy)

			self.oracle.update_trial(trial.trial_id, {'val_accuracy': np.mean(val_accuracy_list)})
			self.save_model(trial.trial_id, model)

    
## initialize the Tuner and search for best hyperparameters
tuner = CVTuner(oracle=kt.oracles.BayesianOptimization(objective='val_accuracy',max_trials=20), hypermodel=build_model)

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10)

tuner.search(x_train, y_train, callbacks=[stop_early])

best_model = tuner.get_best_models()[0]
best_hyperparameters = tuner.get_best_hyperparameters()[0]

best_model.save('/Users/mticli/Documents/BOCCONI/FINAL PROJECT/CNN_LSTM-stock-prices-prediction/algo/SAVED_MODELS/model_0')

## find performance on out of sample data
loss, _, auc = best_model.evaluate(features_test, labels_test, batch_size = best_hyperparameters['batch_size'])

## calculate predictions
y_prob = best_model.predict(features_test)

y_pred = np.argmax(y_prob, axis=1)
y_true = np.argmax(labels_test, axis=1)

## calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
cohen_kappa = cohen_kappa_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred, labels=[0,1])

print('')
print(f'Auc: %{auc*100}')
print(f'Accuracy: %{accuracy*100}')
print(f'Precision: %{precision*100}')
print(f'Recall: %{recall*100}')
print(f'f1_score: %{f1*100}')
print(f'Cohen_kappa: %{cohen_kappa*100}')
print(f'Loss: %{loss*100}')
print('')
print(conf_matrix)
print('')
