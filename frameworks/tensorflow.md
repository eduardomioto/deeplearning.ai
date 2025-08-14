### TensorFlow
TensorFlow is a comprehensive, open-source platform for machine learning. It provides a complete ecosystem of tools, libraries, and community resources.



**Quick Install:**
```bash
python3 -m pip install 'tensorflow[and-cuda]'
```

**Verify GPU Support:**
```bash
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```