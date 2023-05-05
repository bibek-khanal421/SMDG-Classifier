import tensorflow as tf
import numpy as np
import cv2

image_size = 256
num_classes = 1

class GradCAM:
  def __init__(self, model, layerName):
    self.model = model
    self.layerName = layerName
    
    self.gradModel = tf.keras.models.Model(inputs=[self.model.inputs], 
                                           outputs=[self.model.get_layer(self.layerName).output, self.model.output])
    
  def compute_heatmap(self, image, classIdx, eps=1e-8):
    with tf.GradientTape() as tape:
      tape.watch(self.gradModel.get_layer(self.layerName).variables)
      inputs = tf.cast(image, tf.float32)
      (convOutputs, predictions) = self.gradModel(inputs)

      if len(predictions)==1:
        # Binary Classification
        loss = predictions[0]
      else:
        loss = predictions[:, classIdx]
    
    grads = tape.gradient(loss, convOutputs)
    
    castConvOutputs = tf.cast(convOutputs > 0, "float32")
    castGrads = tf.cast(grads > 0, "float32")
    guidedGrads = castConvOutputs * castGrads 
    
    convOutputs = convOutputs[0]
    guidedGrads = guidedGrads[0]
    
    weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
    cam = tf.keras.layers.ReLU()(tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1))
    
    (w, h) = (image.shape[2], image.shape[1])
    heatmap = cv2.resize(cam.numpy(), (w, h))
    
    numer = heatmap - np.min(heatmap)
    denom = (heatmap.max() - heatmap.min()) + eps
    heatmap = numer / denom
    heatmap = (heatmap * 255).astype("uint8")
    return heatmap
    
  def overlay_heatmap(self, heatmap, image, alpha=0.5, colormap=cv2.COLORMAP_JET):
    heatmap = cv2.applyColorMap(heatmap, colormap)
    output = cv2.addWeighted(image, alpha , heatmap, 1 - alpha, 0)
    return (heatmap, output)


def GRADgen(model, layer_name, image):
    gradcam = GradCAM(model, layer_name)
    image = image.reshape(-1, image_size, image_size, 3)
    preds = model.predict(image)
    i = np.argmax(preds[0])
    heatmap = gradcam.compute_heatmap(image, i)
    image = image.reshape(image.shape[1:])
    image = image*255
    image = image.astype(np.uint8)
    heatmap = cv2.resize(heatmap, (image_size, image_size))
    (heatmap, output) = gradcam.overlay_heatmap(heatmap, image, alpha=0.5)
    return output, image, np.round(preds)
