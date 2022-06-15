import tensorflow as tf

@tf.custom_gradient
def threshold_weights(weights):
    bin_w = tf.where(weights < 0,
                0*tf.ones_like(weights, dtype=tf.float32),
                1*tf.ones_like(weights, dtype=tf.float32))
    def grad(dw):
        return dw
    return bin_w, grad

"""
Custom constraint object for getting binary weights. Not suggested to use with
Keras' Conv2D layers for it directly affects the actual weights instead of behaving
as an intermediate process in gradient descent.
"""
class Binarize(tf.keras.constraints.Constraint):
    def __init__(self, threshold=0, units=[0, 1]):
        super(Binarize, self).__init__()
        self.threshold = threshold
        self.min = units[0]
        self.max = units[1]
    def __call__(self, w):
        w_shift = w - self.threshold
        binar = threshold_weights(w_shift)
        return tf.stop_gradient(binar*(self.max-self.min)+ self.min)
    def get_config(self):
        return {'units': self.units,
                'threshold': self.threshold}

# Custom binary fully connected layer.
class BinaryDense(tf.keras.layers.Layer):
    def __init__(self, num_neurons, threshold=0, units=[0,1]):
        super(BinaryDense, self).__init__()
        self.num_neurons = num_neurons
        self.threshold = threshold
        self.min = units[0]
        self.max = units[1]
        
    def build(self, input_shape):
        w_init = tf.random_uniform_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_shape[-1], self.num_neurons), dtype="float32"),
            trainable=True)

    def call(self, input_tensor):
        input_tensor = tf.cast(input_tensor, tf.float32)
        w_shift = self.w - self.threshold
        w_bin = threshold_weights(w_shift)
        w_bin = tf.stop_gradient(w_bin*(self.max-self.min)+ self.min)
        return tf.matmul(input_tensor, w_bin)

# Custom binary convolutional layer.
class BinaryConv2D(tf.keras.layers.Layer):
    def __init__(self, kernel_num, kernel_size, strides=(1,1), padding="VALID",
                 threshold=0, units=[0,1]):
        super(BinaryConv2D, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.threshold = threshold
        self.min = units[0]
        self.max = units[1]
        
    def build(self, input_shape):
        # input_shape=[number_of_images, image_height, image_width, image_channel]
        w_init = tf.random_uniform_initializer()
        # w_shape = [kernel_height, kernel_width, input_channel, number_of_filters]
        w_shape = [self.kernel_size[0], self.kernel_size[1], input_shape[-1], self.kernel_num]
        self.w = tf.Variable(initial_value=w_init(w_shape, dtype=tf.float32),
                        trainable=True)
    def call(self, input_tensor):
        input_tensor = tf.cast(input_tensor, tf.float32)
        # Shifting weights for applying threshold values
        w_shift = self.w - self.threshold
        w_bin = threshold_weights(w_shift)
        w_bin = w_bin * (self.max - self.min) + self.min
        conv = tf.nn.conv2d(input_tensor, w_bin, strides=self.strides, padding=self.padding)
        return conv

# Custom binary filter
class BinaryFilter(tf.keras.layers.Layer):
    def __init__(self):
        super(BinaryFilter, self).__init__()
        
    def call(self):
        pass
