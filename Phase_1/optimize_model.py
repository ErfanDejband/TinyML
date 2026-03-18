import os
import tempfile
import numpy as np
import tensorflow as tf
import tf_keras as keras
import tensorflow_model_optimization as tfmot
from typing import Dict, Any, List

def get_gzipped_model_size(file_path: str) -> float:
    """Returns the gzipped size of a file in kilobytes."""
    import zipfile
    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(file_path)
    return os.path.getsize(zipped_file) / 1024

def optimize_and_convert_model(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    total_training_steps: int|None = None,
    pruning_params: Dict[str, Any] = None,
    quantize: bool = True,
    optimizer: str = 'adam',
    loss: str = 'sparse_categorical_crossentropy',
    metrics: List[str] = ['accuracy']
) -> bytes:
    """
    Applies pruning and quantization to a trained Keras model and converts it to TFLite.

    Args:
        model (keras.Model): The trained Keras model.
        X_train (np.ndarray): The training data, needed for fine-tuning and quantization.
        y_train (np.ndarray): The training labels, needed for fine-tuning the pruned model.
        pruning_params (Dict[str, Any], optional): Parameters for pruning. 
            Defaults to a simple schedule.
        quantize (bool): Whether to apply post-training integer quantization.

    Returns:
        bytes: The converted and optimized TFLite model as a byte array.
    """
    # --- 1. Pruning ---
    if total_training_steps is not None and total_training_steps < len(X_train) * 0.8:
        end_step = int(total_training_steps*0.9) 
    else:
        end_step = int(len(X_train) * 0.8)
    if pruning_params is None:
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.0,   # Start with 0% pruning
                final_sparsity=0.5,     # End with 50% pruning
                begin_step=0,
                end_step=end_step,
                power=3                 # Controls the curve shape
            )
        }

    model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

    # Re-compile the model for fine-tuning
    model_for_pruning.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    print("Fine-tuning the pruned model...")
    # Fine-tune for a few epochs
    model_for_pruning.fit(
        X_train,
        y_train,
        validation_split=0.2, 
        epochs=5,
        callbacks=[tfmot.sparsity.keras.UpdatePruningStep()]
    )
    
    # Strip the pruning wrappers to get the final sparse model
    model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
    print("\nPruning complete.")

    # --- 2. Quantization and TFLite Conversion ---
    converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
    
    if quantize:
        print("Applying post-training integer quantization...")
        # Define a representative dataset generator
        def representative_dataset_gen():
            for i in range(100):
                # Use a small, random subset of the training data
                yield [X_train[i:i+1].astype(np.float32)]

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen
        # Enforce integer-only quantization
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    print("\nQuantization and TFLite conversion complete.")
    
    return tflite_model

if __name__ == '__main__':
    from create_model import create_model
    from prepare_data_for_training import prepare_data

    # --- Setup: Create and train a dummy model ---
    print("--- Preparing Data and Training Initial Model ---")
    data_dir = 'RowData'
    X_train, X_test, y_train, y_test = prepare_data(data_dir, window_size_s=1.0)
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(np.unique(y_train))
    
    base_model = create_model(input_shape=input_shape, num_classes=num_classes)
    print("Training base model...")
    base_model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=0)

    # --- Define a directory to save our models ---
    output_dir = 'models'
    os.makedirs(output_dir, exist_ok=True) # Create the directory if it doesn't exist
    
    # Save the base model to compare size
    base_model_file = os.path.join(output_dir, 'base_model.h5')
    base_model.save(base_model_file, save_format='h5', include_optimizer=False)
    print(f"Base model saved to: {base_model_file}")
    print(f"Base model size: {get_gzipped_model_size(base_model_file):.2f} KB")

    # --- Optimization ---
    print("\n--- Optimizing Model ---")
    tflite_model_quant = optimize_and_convert_model(base_model, X_train, y_train)

    # --- Save and Compare ---
    tflite_model_path = os.path.join(output_dir, 'magic_wand_model.tflite')
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model_quant)
        
    print(f"Optimized (quantized) TFLite model size: {get_gzipped_model_size(tflite_model_path):.2f} KB")
    print(f"TFLite model saved to: {tflite_model_path}")
