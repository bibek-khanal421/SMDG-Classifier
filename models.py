import tensorflow as tf
class ResNet50():
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
    
    def build_model(self):
        tf.keras.backend.clear_session()
        inputs = tf.keras.Input(shape=self.input_shape)
        resnet = tf.keras.applications.ResNet50(input_shape = self.input_shape,
                                                 include_top=False, 
                                                 input_tensor=inputs,weights='imagenet')
        x = resnet.get_layer('conv5_block3_out').output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(self.num_classes, activation='sigmoid')(x)
        model = tf.keras.Model(inputs, outputs)
        return model

class Xception():
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
    
    def build_model(self):
        tf.keras.backend.clear_session()
        inputs = tf.keras.Input(shape=self.input_shape)
        xception = tf.keras.applications.Xception(input_shape = self.input_shape,
                                                 include_top=False, 
                                                 input_tensor=inputs,
                                                 weights='imagenet')
        x = xception.get_layer('block14_sepconv2_act').output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(self.num_classes, activation='sigmoid')(x)
        model = tf.keras.Model(inputs, outputs)
        return model
