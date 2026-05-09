import os
import tensorflow as tf

def build_and_save_classifier():
    print("==================================================")
    print(" EFFICIENTNET-B0 CLASSIFIER SURGERY")
    print("==================================================")

    model_path = 'models/hybrid_7ch/Eff_7ch.keras'
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Could not find {model_path}. Make sure you are in the PBL directory.")
        return

    print(f"\n[1] Loading raw expanded backbone from {model_path}...")
    # Load without compiled metrics since it's just a backbone
    model_eff = tf.keras.models.load_model(model_path, compile=False)

    print("[2] Attaching Deepfake Classification Head...")
    # The backbone outputs a spatial feature map: (None, 8, 8, 1280)
    # We pool it to a 1D vector of 1280 features
    x = tf.keras.layers.GlobalAveragePooling2D(name="deepfake_gap")(model_eff.output)
    
    # Add dropout to prevent overfitting during training
    x = tf.keras.layers.Dropout(0.4, name="deepfake_dropout")(x)
    
    # Add the single neuron Sigmoid classification head (1 = Real, 0 = Fake)
    output = tf.keras.layers.Dense(1, activation='sigmoid', name="deepfake_classifier")(x)

    # 3. Create the final compilable model
    trainable_eff = tf.keras.Model(inputs=model_eff.input, outputs=output, name="EfficientNetB0_7CH_Classifier")

    print("[3] Compiling model with Adam & Binary Crossentropy...")
    trainable_eff.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print("\n[INFO] Model Architecture Summary:")
    trainable_eff.summary()

    # 4. Save the finalized trainable model
    save_path = 'models/hybrid_7ch/Eff_7ch_Trainable.keras'
    print(f"\n[4] Saving fully trainable classifier to: {save_path}")
    trainable_eff.save(save_path)
    print("DONE! EfficientNet is now ready to be fine-tuned on the Deepfake dataset!")

if __name__ == "__main__":
    build_and_save_classifier()
