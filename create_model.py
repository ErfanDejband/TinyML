from typing import List, Tuple
import tf_keras as keras
from tf_keras import layers

def create_model(
    input_shape: Tuple[int, int],
    num_classes: int,
    conv_filters: int = 8,
    conv_kernel_size: int = 3,
    dense_layers: List[int] = [16],
    activation: str = 'relu',
    optimizer: str = 'adam',
    loss: str = 'sparse_categorical_crossentropy',
    metrics: List[str] = ['accuracy']
) -> keras.Model:
    """
    Creates and compiles a Convolutional Neural Network (CNN) model for time-series data.

    Args:
        input_shape (Tuple[int, int]): The shape of the input data (timesteps, features).
        num_classes (int): The number of output classes.
        conv_filters (int): Number of filters in the Conv1D layer.
        conv_kernel_size (int): Kernel size for the Conv1D layer.
        dense_layers (List[int]): A list of integers for the units in each dense layer.
        activation (str): The activation function for the dense layers.
        optimizer (str): The optimizer to use for compiling the model.
        loss (str): The loss function to use.
        metrics (List[str]): The metrics to evaluate.

    Returns:
        keras.Model: The compiled TensorFlow model.
    """
    model_layers = [
        layers.Conv1D(
            filters=conv_filters,
            kernel_size=conv_kernel_size,
            activation=activation,
            input_shape=input_shape
        ),
        layers.MaxPooling1D(2),
        layers.Flatten()
    ]
    
    for units in dense_layers:
        model_layers.append(layers.Dense(units, activation=activation))
        
    model_layers.append(layers.Dense(num_classes, activation='softmax'))
    
    model = keras.Sequential(model_layers)
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    return model

if __name__ == "__main__":
    # Example usage based on the new data shape (e.g., 50 timesteps, 3 features)
    # This shape comes from prepare_data_for_training.py
    INPUT_SHAPE = (50, 3) 
    NUM_CLASSES = 2       # "Wave" or "Idle"

    # Create the model
    my_model = create_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)

    # Print the model summary
    print("CNN Model created successfully!")
    my_model.summary()
