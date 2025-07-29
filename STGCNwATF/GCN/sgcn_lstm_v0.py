from keras.layers import Lambda
from keras.layers import LeakyReLU, Add, Activation
import pdb

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dropout, Dense, Input, LSTM, concatenate, ConvLSTM2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import *
from IPython.core.debugger import set_trace
from tensorflow.keras import initializers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.callbacks import Callback


class Sgcn_Lstm():
    def __init__(self, train_x, train_y, AD, AD2, bias_mat_1, bias_mat_2, lr=0.0001, epoach=200, batch_size=10):
        self.train_x = train_x
        self.train_y = train_y
        self.AD = AD
        self.AD2 = AD2
        #self.bias_mat_1 = bias_mat_1
        #self.bias_mat_2 = bias_mat_2
        # to make the relationshsip between the joints learnable
        self.bias_mat_1 = tf.Variable(initial_value=bias_mat_1, trainable=True, dtype=tf.float32)
        self.bias_mat_2 = tf.Variable(initial_value=bias_mat_2, trainable=True, dtype=tf.float32)
        ###
        self.lr = lr
        self.epoach =epoach
        self.batch_size = batch_size
        self.num_joints = 25

    def sgcn(self, Input):
        """Temporal convolution"""
        k1 = tf.keras.layers.Conv2D(64, (9,1), padding='same', activation='relu')(Input)
        k =  concatenate([Input, k1], axis=-1)
        """Graph Convolution"""
        
        """first hop localization"""
        x1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1,1), strides=1, activation='relu')(k)

        #start Jubran
        expand_layer = Lambda(lambda x: tf.expand_dims(x, axis=3))
        expand_x1 = expand_layer(x1)
        # end jubran
        #expand_x1 = tf.expand_dims(x1, axis=3) #comented jubran - compatability

        f_1 = ConvLSTM2D(filters=25, kernel_size=(1,1), input_shape=(None,None,25,1,3), return_sequences=True)(expand_x1)
        f_1 = f_1[:,:,:,0,:]
        logits = f_1
        #start Jubran
        #leaky = LeakyReLU(alpha=0.2)(logits)  # Keras-friendly
        #bias_corrected = tf.reshape(self.bias_mat_1, (1, 1, 25, 25))
        #added = Add()([leaky, bias_corrected])
        #coefs = Activation('softmax')(added)
        # added to make the relationshsip between the joints learanable
        coefs = tf.keras.layers.Lambda(lambda x: tf.nn.softmax(tf.nn.leaky_relu(x[0]) + x[1]))([logits, self.bias_mat_1])

        #end jubran
        #coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + self.bias_mat_1)

        gcn_x1 = tf.keras.layers.Lambda(lambda x: tf.einsum('ntvw,ntwc->ntvc', x[0], x[1]))([coefs, x1])
       
        """second hop localization"""
        y1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1,1), strides=1, activation='relu')(k)

        #start Jubran
        expand_layer = Lambda(lambda y: tf.expand_dims(y, axis=3))
        expand_y1 = expand_layer(y1)
        # end jubran
        #expand_y1 = tf.expand_dims(y1, axis=3)

        f_2 = ConvLSTM2D(filters=25, kernel_size=(1,1), input_shape=(None,None,25,1,3), return_sequences=True)(expand_y1)
        f_2 = f_2[:,:,:,0,:]
        logits = f_2
        #start Jubran
        #leaky = LeakyReLU(alpha=0.2)(logits)  # Keras-friendly
        #bias_corrected = tf.reshape(self.bias_mat_1, (1, 1, 25, 25))
        #added = Add()([leaky, bias_corrected])
        #coefs = Activation('softmax')(added)
        # added to make the relationshsip between the joints learanable
        coefs = tf.keras.layers.Lambda(lambda x: tf.nn.softmax(tf.nn.leaky_relu(x[0]) + x[1]))([logits, self.bias_mat_2])
        #end jubran
        #coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + self.bias_mat_2)

        gcn_y1 = tf.keras.layers.Lambda(lambda x: tf.einsum('ntvw,ntwc->ntvc', x[0], x[1]))([coefs, y1])

        gcn_1 = concatenate([gcn_x1, gcn_y1], axis=-1)
        
        """Temporal convolution"""
        z1 = tf.keras.layers.Conv2D(16, (9,1), padding='same', activation='relu')(gcn_1)
        z1 = Dropout(0.25)(z1)
        z2 = tf.keras.layers.Conv2D(16, (15,1), padding='same', activation='relu')(z1)
        z2 = Dropout(0.25)(z2)
        z3 = tf.keras.layers.Conv2D(16, (20,1), padding='same', activation='relu')(z2)
        z3 = Dropout(0.25)(z3)
        z = concatenate([z1, z2, z3], axis=-1) 
        return z

    def Lstm(self,x):
        x = tf.keras.layers.Reshape(target_shape=(-1,x.shape[2]*x.shape[3]))(x)
        rec = LSTM(80, return_sequences=True)(x)
        rec = Dropout(0.25)(rec)
        rec1 = LSTM(40, return_sequences=True)(rec)
        rec1 = Dropout(0.25)(rec1)
        rec2 = LSTM(40, return_sequences=True)(rec1)
        rec2 = Dropout(0.25)(rec2)
        rec3 = LSTM(80)(rec2)
        rec3 = Dropout(0.25)(rec3)
        out = Dense(1, activation = 'linear')(rec3)
        return out
      
    def train(self):
        seq_input = Input(shape=(None, self.train_x.shape[2], self.train_x.shape[3]), batch_size=None)
        x = self.sgcn(seq_input)
        y = self.sgcn(x)
        y = y + x
        z = self.sgcn(y)
        z = z + y
        out = self.Lstm(z)
        self.model = Model(seq_input, out)
        #self.model.compile(loss=tf.keras.losses.Huber(delta=0.1), experimental_steps_per_execution = 50, optimizer= tf.keras.optimizers.Adam(learning_rate=self.lr))
        #Jubran
        self.model.compile(loss=tf.keras.losses.Huber(delta=0.1), optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr))
        checkpoint = ModelCheckpoint("best model ex4/best_model.keras", monitor='val_loss', save_best_only=True, mode='auto')
        #history = self.model.fit(self.train_x, self.train_y, validation_split=0.2, epochs=self.epoach, batch_size=self.batch_size, callbacks=[checkpoint])

       # Create the bias matrix monitoring callback
        bias_monitor = BiasMatrixMonitor(
            self.bias_mat_1, 
            self.bias_mat_2, 
            print_frequency=1  # Print every batch (change to higher number for less frequent printing)
        )

        print_callback = BiasPrintCallback(self)
        # Include the bias monitor in callbacks
        callbacks = [checkpoint, bias_monitor]
        history = self.model.fit(self.train_x,self.train_y,validation_split=0.2,epochs=self.epoach,batch_size=self.batch_size,callbacks=callbacks)

        return history
      
    def prediction(self, data):
        y_pred = self.model.predict(data)
        return y_pred

class BiasPrintCallback(tf.keras.callbacks.Callback):
    def __init__(self, model_or_wrapper):
        super().__init__()
        self.wrapper = model_or_wrapper

    def on_train_batch_end(self, batch, logs=None):
        # Print biases
        b1 = self.wrapper.bias_mat_1.numpy()
        b2 = self.wrapper.bias_mat_2.numpy()
        print(f"[Batch {batch}] bias_mat_1 mean={b1.mean():.6f}, bias_mat_2 mean={b2.mean():.6f}")

        # Optionally print all trainable variables (summary)
        #for var in self.wrapper.model.trainable_variables:
        #    print(f"{var.name}: mean={tf.reduce_mean(var).numpy():.6f}")



class BiasMatrixMonitor(Callback):
    def __init__(self, bias_mat_1, bias_mat_2, print_frequency=1):
        super(BiasMatrixMonitor, self).__init__()
        self.bias_mat_1 = bias_mat_1
        self.bias_mat_2 = bias_mat_2
        self.print_frequency = print_frequency
        self.batch_count = 0
        
    def on_batch_end(self, batch, logs=None):
        self.batch_count += 1
        
        # Print every 'print_frequency' batches
        if self.batch_count % self.print_frequency == 0:
            print(f"\n=== Batch {self.batch_count} ===")
            print("bias_mat_1 (first 5x5 corner):")
            print(self.bias_mat_1.numpy()[:5, :5])
            print("bias_mat_2 (first 5x5 corner):")
            print(self.bias_mat_2.numpy()[:5, :5])
            print("=" * 30)

# Alternative: More detailed monitoring with statistics
class DetailedBiasMatrixMonitor(Callback):
    def __init__(self, bias_mat_1, bias_mat_2, print_frequency=10):
        super(DetailedBiasMatrixMonitor, self).__init__()
        self.bias_mat_1 = bias_mat_1
        self.bias_mat_2 = bias_mat_2
        self.print_frequency = print_frequency
        self.batch_count = 0
        self.initial_bias_1 = None
        self.initial_bias_2 = None
        
    def on_train_begin(self, logs=None):
        # Store initial values for comparison
        self.initial_bias_1 = self.bias_mat_1.numpy().copy()
        self.initial_bias_2 = self.bias_mat_2.numpy().copy()
        
    def on_batch_end(self, batch, logs=None):
        self.batch_count += 1
        
        if self.batch_count % self.print_frequency == 0:
            current_bias_1 = self.bias_mat_1.numpy()
            current_bias_2 = self.bias_mat_2.numpy()
            
            # Calculate change from initial values
            change_1 = np.abs(current_bias_1 - self.initial_bias_1)
            change_2 = np.abs(current_bias_2 - self.initial_bias_2)
            
            print(f"\n=== Batch {self.batch_count} Bias Matrix Statistics ===")
            print(f"bias_mat_1 - Mean: {current_bias_1.mean():.6f}, Std: {current_bias_1.std():.6f}")
            print(f"bias_mat_1 - Max change from initial: {change_1.max():.6f}")
            print(f"bias_mat_2 - Mean: {current_bias_2.mean():.6f}, Std: {current_bias_2.std():.6f}")
            print(f"bias_mat_2 - Max change from initial: {change_2.max():.6f}")
            
            # Show a small sample of actual values
            print("bias_mat_1 sample (top-left 3x3):")
            print(current_bias_1[:3, :3])
            print("bias_mat_2 sample (top-left 3x3):")
            print(current_bias_2[:3, :3])
            print("=" * 50)
