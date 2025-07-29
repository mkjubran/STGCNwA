from keras.layers import Lambda, Layer
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

# Custom layer to make bias matrices learnable
class LearnableAttentionLayer(Layer):
    def __init__(self, bias_matrix_init, name_suffix="", **kwargs):
        super(LearnableAttentionLayer, self).__init__(**kwargs)
        # Convert numpy array to list for serialization
        if hasattr(bias_matrix_init, 'numpy'):
            self.bias_matrix_init = bias_matrix_init.numpy().tolist()
        elif isinstance(bias_matrix_init, np.ndarray):
            self.bias_matrix_init = bias_matrix_init.tolist()
        else:
            self.bias_matrix_init = bias_matrix_init
        self.name_suffix = name_suffix
        self.bias_matrix_shape = np.array(self.bias_matrix_init).shape
        
    def build(self, input_shape):
        # Create the learnable bias matrix as a layer weight
        self.bias_matrix = self.add_weight(
            name=f'attention_bias_{self.name_suffix}',
            shape=self.bias_matrix_shape,
            initializer='zeros',  # Start from zeros, will be set in call
            trainable=True
        )
        # Set initial values
        initial_values = tf.constant(self.bias_matrix_init, dtype=tf.float32)
        self.bias_matrix.assign(initial_values)
        super(LearnableAttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        logits = inputs
        # Apply learnable attention
        attention_weights = tf.nn.softmax(tf.nn.leaky_relu(logits) + self.bias_matrix)
        return attention_weights
    
    def get_config(self):
        config = super(LearnableAttentionLayer, self).get_config()
        config.update({
            "bias_matrix_init": self.bias_matrix_init,
            "name_suffix": self.name_suffix,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class Sgcn_Lstm():
    def __init__(self, train_x, train_y, AD, AD2, bias_mat_1, bias_mat_2, lr=0.0001, epoach=200, batch_size=10):
        self.train_x = train_x
        self.train_y = train_y
        self.AD = AD
        self.AD2 = AD2
        
        # Store initial bias matrices for the custom layers
        self.bias_mat_1_init = bias_mat_1.numpy() if hasattr(bias_mat_1, 'numpy') else bias_mat_1
        self.bias_mat_2_init = bias_mat_2.numpy() if hasattr(bias_mat_2, 'numpy') else bias_mat_2
        
        self.lr = lr
        self.epoach = epoach
        self.batch_size = batch_size
        self.num_joints = 25
        
        # These will be set during model building
        self.attention_layer_1 = None
        self.attention_layer_2 = None

    def sgcn(self, Input):
        """Temporal convolution"""
        k1 = tf.keras.layers.Conv2D(64, (9,1), padding='same', activation='relu')(Input)
        k = concatenate([Input, k1], axis=-1)
        
        """Graph Convolution"""
        
        """first hop localization with learnable attention"""
        x1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1,1), strides=1, activation='relu')(k)
        expand_layer = Lambda(lambda x: tf.expand_dims(x, axis=3))
        expand_x1 = expand_layer(x1)
        
        f_1 = ConvLSTM2D(filters=25, kernel_size=(1,1), input_shape=(None,None,25,1,3), return_sequences=True)(expand_x1)
        f_1 = f_1[:,:,:,0,:]
        logits = f_1
        
        # Use learnable attention layer
        if self.attention_layer_1 is None:
            self.attention_layer_1 = LearnableAttentionLayer(self.bias_mat_1_init, name_suffix="1")
        coefs = self.attention_layer_1(logits)
        
        gcn_x1 = tf.keras.layers.Lambda(lambda x: tf.einsum('ntvw,ntwc->ntvc', x[0], x[1]))([coefs, x1])
        
        """second hop localization with learnable attention"""
        y1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1,1), strides=1, activation='relu')(k)
        expand_layer = Lambda(lambda y: tf.expand_dims(y, axis=3))
        expand_y1 = expand_layer(y1)
        
        f_2 = ConvLSTM2D(filters=25, kernel_size=(1,1), input_shape=(None,None,25,1,3), return_sequences=True)(expand_y1)
        f_2 = f_2[:,:,:,0,:]
        logits = f_2
        
        # Use learnable attention layer for second hop
        if self.attention_layer_2 is None:
            self.attention_layer_2 = LearnableAttentionLayer(self.bias_mat_2_init, name_suffix="2")
        coefs = self.attention_layer_2(logits)
        
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

    def Lstm(self, x):
        x = tf.keras.layers.Reshape(target_shape=(-1, x.shape[2]*x.shape[3]))(x)
        rec = LSTM(80, return_sequences=True)(x)
        rec = Dropout(0.25)(rec)
        rec1 = LSTM(40, return_sequences=True)(rec)
        rec1 = Dropout(0.25)(rec1)
        rec2 = LSTM(40, return_sequences=True)(rec1)
        rec2 = Dropout(0.25)(rec2)
        rec3 = LSTM(80)(rec2)
        rec3 = Dropout(0.25)(rec3)
        out = Dense(1, activation='linear')(rec3)
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
        self.model.compile(loss=tf.keras.losses.Huber(delta=0.1), optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr))
        
        # Custom callback to monitor the learnable attention weights
        class AttentionMonitor(tf.keras.callbacks.Callback):
            def __init__(self, model_instance):
                self.model_instance = model_instance
                self.batch_count = 0
                
            def on_batch_end(self, batch, logs=None):
                self.batch_count += 1
                if self.batch_count % 10 == 0:  # Print every 10 batches
                    print(f"\n=== Batch {self.batch_count} - Learnable Attention Weights ===")
                    
                    # Get attention weights from the custom layers
                    for layer in self.model.layers:
                        if isinstance(layer, LearnableAttentionLayer):
                            weights = layer.bias_matrix.numpy()
                            print(f"Layer {layer.name_suffix} attention weights (3x3 sample):")
                            print(weights[:3, :3])
                    print("=" * 50)
        
        attention_monitor = AttentionMonitor(self)
        checkpoint = ModelCheckpoint("best model ex4/best_model.keras", monitor='val_loss', save_best_only=True, mode='auto')
        
        #callbacks = [checkpoint, attention_monitor]
        callbacks = [checkpoint]
        
        history = self.model.fit(
            self.train_x, 
            self.train_y, 
            validation_split=0.2, 
            epochs=self.epoach, 
            batch_size=self.batch_size, 
            callbacks=callbacks
        )
        return history

    def prediction(self, data):
        y_pred = self.model.predict(data)
        return y_pred
