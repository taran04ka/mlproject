import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.sparsity.keras import UpdatePruningStep

class ModelBuilder:
    def build_model(self, optimizer='adam', activation='relu', dropout_rate=0.0):
        with tf.device('/GPU:0'):
            model = Sequential()
            model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Flatten())
            model.add(Dense(128, activation=activation))
            model.add(Dropout(dropout_rate))
            model.add(Dense(10, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    def build_pruned_model(self, optimizer='adam', activation='relu', dropout_rate=0.0):
        model = self.build_model(optimizer, activation, dropout_rate)
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                                     final_sparsity=0.90,
                                                                     begin_step=0,
                                                                     end_step=2000)
        }
        model_pruned = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
        model_pruned.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model_pruned

    def train_model(self, model, X_train, y_train, X_val, y_val, epochs=3, batch_size=32):
        callbacks = [
            UpdatePruningStep(),
        ]
        history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                            validation_data=(X_val, y_val), callbacks=callbacks)
        return history
