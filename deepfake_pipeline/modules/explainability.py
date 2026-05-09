import tensorflow as tf
import numpy as np
import cv2

class MultiResolutionGradCAM:
    def __init__(self, model):
        self.model = model

    def get_last_conv_layer_name(self, model_type):
        """Standardized layer names based on common model implementations."""
        if 'xception' in model_type:
            return 'block14_sepconv2_act'
        elif 'resnet50' in model_type:
            return 'conv5_block3_out'
        elif 'efficientnet' in model_type:
            return 'top_conv'
        return None

    def compute_heatmap(self, img_array, layer_name):
        """Basic Grad-CAM heatmap generation."""
        grad_model = tf.keras.models.Model(
            [self.model.inputs], [self.model.get_layer(layer_name).output, self.model.output]
        )

        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            # Find the top class index (usually Real/Fake is at index 0 or 1)
            # In the user's notebook, Xception/ResNet were binary (index 0), EfficientNet was categorical (index 1)
            class_idx = tf.argmax(preds[0])
            class_channel = preds[:, class_idx]

        # 1. Gradients of the top class with respect to the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)

        # 2. Global Average Pooling of the gradients (alphas)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # 3. Weighted activation map (Fusion)
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # 4. Normalize (ReLU + Normalize to 0-1)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()

    def generate_multi_res_cam(self, img_array, model_type):
        """
        Multi-Resolution CAM:
        Fuses heatmaps from the final Conv layer and a mid-level Conv layer.
        """
        # 1. Identify layers
        last_layer = self.get_last_conv_layer_name(model_type)
        # Mid-level layer (to capture texture patterns/GAN noise)
        if 'xception' in model_type: mid_layer = 'block10_sepconv2_act'
        elif 'resnet50' in model_type: mid_layer = 'conv4_block6_out'
        elif 'efficientnet' in model_type: mid_layer = 'block6a_expand_activation'
        else: mid_layer = last_layer # Fallback
        
        try:
            h1 = self.compute_heatmap(img_array, last_layer)
            h2 = self.compute_heatmap(img_array, mid_layer)
            
            # Weighted average fusion (e.g., 60% top level, 40% mid level)
            # They need to be resized to the same shape if they're different (unlikely though)
            h1_res = cv2.resize(h1, (224, 224)) if h1.shape[0] != 224 else h1
            h2_res = cv2.resize(h2, h1_res.shape[::-1])
            
            fused_heatmap = 0.6 * h1_res + 0.4 * h2_res
            return fused_heatmap / np.max(fused_heatmap)
            
        except Exception as e:
            print(f"Explainability Error: {e}")
            return None

    def overlay_on_image(self, img_array, heatmap, intensity=0.5):
        """Overlap heatmap on original RGB image."""
        # 1. Image preprocessing (un-normalize if needed, ensure [0, 255] uint8)
        if np.max(img_array) <= 1.0: img_array = img_array * 255.0
        img = img_array.squeeze().astype(np.uint8)
        
        # 2. Rescale heatmap and apply color map
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        color_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # 3. Superimpose
        superimposed_img = color_heatmap * intensity + img
        return np.clip(superimposed_img, 0, 255).astype(np.uint8)
