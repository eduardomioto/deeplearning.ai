# TensorFlow - Complete Guide

Google's open-source machine learning framework for building and deploying AI models at scale.

## 📚 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Model Building](#model-building)
- [Training & Evaluation](#training--evaluation)
- [Deployment](#deployment)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Resources](#resources)

## 🎯 Overview

TensorFlow is an end-to-end open-source platform for machine learning. It has a comprehensive, flexible ecosystem of tools, libraries, and community resources that lets researchers push the state-of-the-art in ML and developers easily build and deploy ML-powered applications.

## ✨ Key Features

- **Multi-platform support** - Run on CPU, GPU, TPU, mobile, and edge devices
- **Production ready** - Deploy models in production with TensorFlow Serving
- **Flexible architecture** - Build models using Keras, Estimators, or custom layers
- **Visualization tools** - TensorBoard for model visualization and debugging
- **Large ecosystem** - Extensive libraries for computer vision, NLP, and more
- **Community support** - Large, active community and comprehensive documentation

## 🚀 Installation

### Prerequisites
- Python 3.7-3.11 (3.12 support coming soon)
- pip package manager
- Optional: CUDA for GPU support

### Basic Installation

**CPU-only version:**
```bash
pip install tensorflow
```

**GPU version with CUDA support:**
```bash
pip install 'tensorflow[and-cuda]'
```

**Verify installation:**
```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

### GPU Setup (Optional)

**Install CUDA Toolkit:**
```bash
# Ubuntu/Debian
sudo apt-get install nvidia-cuda-toolkit

# Windows
# Download from https://developer.nvidia.com/cuda-downloads
```

**Verify GPU detection:**
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Development Installation

**Install from source:**
```bash
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
pip install -e .
```

## 🚀 Quick Start

### Hello World Example

```python
import tensorflow as tf

# Create a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("TensorFlow version:", tf.__version__)
print("Model created successfully!")
```

### Basic Neural Network

```python
import tensorflow as tf
import numpy as np

# Generate sample data
X = np.random.random((1000, 20))
y = np.random.randint(0, 2, (1000,))

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(20,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile and train
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)
```

## 🧠 Core Concepts

### 1. **Tensors**
Tensors are multi-dimensional arrays that represent data in TensorFlow.

```python
# Scalar (0D tensor)
scalar = tf.constant(42)

# Vector (1D tensor)
vector = tf.constant([1, 2, 3, 4])

# Matrix (2D tensor)
matrix = tf.constant([[1, 2], [3, 4]])

# Higher dimensional tensor
tensor_3d = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
```

### 2. **Graphs and Sessions**
TensorFlow 2.x uses eager execution by default, but you can still use graph mode:

```python
# Eager execution (default in TF 2.x)
@tf.function
def my_function(x, y):
    return x + y

# Graph mode
with tf.Graph().as_default():
    a = tf.constant(5)
    b = tf.constant(3)
    c = a + b
```

### 3. **Variables**
Variables maintain state across graph executions:

```python
# Create a variable
weights = tf.Variable(tf.random.normal([784, 10]))
bias = tf.Variable(tf.zeros([10]))

# Update variable
weights.assign_add(tf.random.normal([784, 10]) * 0.01)
```

## 🏗️ Model Building

### Sequential API (Simplest)

```python
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])
```

### Functional API (Flexible)

```python
inputs = keras.Input(shape=(784,))
x = keras.layers.Dense(128, activation='relu')(inputs)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(64, activation='relu')(x)
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
```

### Subclassing API (Most Flexible)

```python
class MyModel(keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = keras.layers.Dense(128, activation='relu')
        self.dropout = keras.layers.Dropout(0.2)
        self.dense2 = keras.layers.Dense(10, activation='softmax')
    
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dropout(x)
        return self.dense2(x)

model = MyModel()
```

## 🎯 Training & Evaluation

### Data Preparation

```python
# Load and preprocess data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize pixel values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape for dense layers
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)
```

### Model Compilation

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### Training

```python
# Basic training
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Training with callbacks
callbacks = [
    keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2),
    keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
]

history = model.fit(
    x_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks
)
```

### Evaluation

```python
# Evaluate on test set
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f'Test accuracy: {test_accuracy:.4f}')

# Make predictions
predictions = model.predict(x_test[:5])
print(f'Predictions shape: {predictions.shape}')
```

## 🚀 Deployment

### Save and Load Models

```python
# Save model
model.save('my_model.h5')

# Load model
loaded_model = keras.models.load_model('my_model.h5')
```

### TensorFlow Serving

```bash
# Install TensorFlow Serving
pip install tensorflow-serving-api

# Save model in SavedModel format
model.save('saved_model', save_format='tf')

# Start serving (requires Docker)
docker run -p 8501:8501 --mount type=bind,source=/path/to/saved_model,target=/models/my_model -e MODEL_NAME=my_model -t tensorflow/serving
```

### Mobile Deployment

```python
# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save TFLite model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

## 💡 Best Practices

### 1. **Performance Optimization**
- Use `tf.data` for efficient data pipelines
- Enable mixed precision training with `tf.keras.mixed_precision`
- Profile your models with TensorBoard
- Use appropriate batch sizes for your hardware

### 2. **Memory Management**
- Clear GPU memory when needed: `tf.keras.backend.clear_session()`
- Use `tf.function` decorators for graph optimization
- Monitor memory usage with TensorBoard

### 3. **Code Organization**
- Separate data preprocessing, model definition, and training
- Use configuration files for hyperparameters
- Implement proper logging and monitoring
- Version control your experiments

## 🔧 Troubleshooting

### Common Issues

**GPU not detected:**
```bash
# Check CUDA installation
nvidia-smi

# Verify TensorFlow GPU support
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**Memory issues:**
```python
# Limit GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
```

**Version compatibility:**
```bash
# Check Python version compatibility
python --version

# Install specific TensorFlow version
pip install tensorflow==2.10.0
```

## 📚 Resources

### Official Documentation
- [TensorFlow Guide](https://www.tensorflow.org/guide)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [TensorFlow API Reference](https://www.tensorflow.org/api_docs)
- [TensorFlow Examples](https://github.com/tensorflow/examples)

### Learning Resources
- [Deep Learning Specialization](https://www.deeplearning.ai/)
- [TensorFlow YouTube Channel](https://www.youtube.com/user/tensorflow)
- [TensorFlow Blog](https://blog.tensorflow.org/)

### Community & Support
- [TensorFlow Forum](https://discuss.tensorflow.org/)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/tensorflow)
- [GitHub Issues](https://github.com/tensorflow/tensorflow/issues)
- [Reddit r/TensorFlow](https://www.reddit.com/r/TensorFlow/)

### Books & Courses
- "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" by Aurélien Géron
- "Deep Learning with Python" by François Chollet
- [Coursera TensorFlow Specialization](https://www.coursera.org/specializations/tensorflow-in-practice)

---

**Happy TensorFlow-ing! 🚀✨**

*TensorFlow makes machine learning accessible to everyone. Start building amazing AI applications today!*

