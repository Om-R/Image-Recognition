import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Define the residual block
def residual_block(x, filters):
    shortcut = x  # Save the input for the shortcut connection

    # First convolutional layer
    x = layers.Conv2D(filters, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Second convolutional layer
    x = layers.Conv2D(filters, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)

    # If the number of filters in the shortcut is different, apply a convolution
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, kernel_size=1, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    # Add the shortcut connection
    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)
    return x

# Build the ResNet model
def build_resnet(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Stack of residual blocks
    for _ in range(3):
        x = residual_block(x, 64)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    for _ in range(3):
        x = residual_block(x, 128)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    for _ in range(3):
        x = residual_block(x, 256)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, x)
    return model

# Load and preprocess the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0  # Normalize to [0, 1]
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)  # One-hot encode labels
y_test = to_categorical(y_test, 10)

# Create the model
input_shape = (32, 32, 3)  # Example input shape for CIFAR-10
num_classes = 10  # Number of classes in CIFAR-10
model = build_resnet(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=20, batch_size=64, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc:.4f}')

# Make predictions (optional)
predictions = model.predict(x_test)
predicted_classes = tf.argmax(predictions, axis=1)