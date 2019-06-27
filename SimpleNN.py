from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt


def single_layer(celsius_q, fahrenheit_a):
	model = tf.keras.Sequential([
		tf.keras.layers.Dense(units=1, input_shape=[1])
	])

	model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
	history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)

	plt.xlabel('Epoch Number')
	plt.ylabel("Loss Magnitude")
	plt.plot(history.history['loss'])
	print(model.predict([100.0]))




def main():
	celsius_q    = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
	fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)

	single_layer(celsius_q, fahrenheit_a)


if __name__ == '__main__':
	main()