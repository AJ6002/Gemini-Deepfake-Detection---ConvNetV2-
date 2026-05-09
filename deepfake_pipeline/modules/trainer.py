import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

class MultiStageTrainer:
    def __init__(self, datagen_params, models_dict):
        self.datagen_params = datagen_params
        self.models = models_dict

    def stage_1_train_base_models(self, train_dir, val_dir, epochs=10):
        """
        Stage 1: Standard training of individual backbones.
        Focus: Maximizing diversity and recall.
        """
        train_gen = tf.keras.preprocessing.image.ImageDataGenerator(**self.datagen_params)
        val_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

        for name, model in self.models.items():
            print(f"--- Stage 1: Fine-tuning {name} ---")
            train_data = train_gen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32)
            val_data = val_gen.flow_from_directory(val_dir, target_size=(224, 224), batch_size=32)
            
            # Use specific callbacks (e.g., EarlyStopping, ReduceLROnPlateau)
            model.fit(train_data, validation_data=val_data, epochs=epochs)
            model.save(f"stage1_{name}.keras")
            
    def stage_2_disjoint_validation(self, val_dir_disjoint):
        """
        Stage 2: Multi-Stage Disjoint Training (Weight Tuning).
        Calculates optimal weights for the hybrid ensemble on a disjoint validation set.
        """
        print("--- Stage 2: Disjoint Validation (Weight Optimization) ---")
        val_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        val_data = val_gen.flow_from_directory(val_dir_disjoint, target_size=(224, 224), batch_size=32, shuffle=False)
        
        y_true = val_data.classes
        model_preds = {}
        
        # 1. Get predictions for each base model on disjoint set
        for name, model in self.models.items():
            print(f"Sampling {name} on disjoint set...")
            y_pred = model.predict(val_data)
            # Normalize to single-class prob
            model_preds[name] = y_pred[:, 1] if y_pred.shape[1] > 1 else y_pred[:, 0]
            
        # 2. Optimize Weights (Simple Grid Search or Accuracy Calculation)
        # In a real environment, we'd use a small MLP but here we'll use Accuracy-based weighting.
        accuracies = {}
        for name, probs in model_preds.items():
            acc = np.mean((probs >= 0.5) == y_true)
            accuracies[name] = acc
            print(f"Disjoint Accuracy ({name}): {acc:.4f}")
            
        # 3. Dynamic Hybrid Weighting logic (Base for EnsembleModel weights)
        # Models with higher disjoint accuracy get higher priority weights.
        return accuracies
