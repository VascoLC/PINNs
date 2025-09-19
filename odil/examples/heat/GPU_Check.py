import tensorflow as tf
import os

print("-" * 50)
print("TensorFlow Version:", tf.__version__)

# Check for CUDA_VISIBLE_DEVICES
cuda_var = os.environ.get('CUDA_VISIBLE_DEVICES')
if cuda_var:
    print(f"CUDA_VISIBLE_DEVICES is set to: '{cuda_var}'")
else:
    print("CUDA_VISIBLE_DEVICES is NOT set.")

print("-" * 50)

gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print(f"✅ SUCCESS: TensorFlow found {len(gpus)} GPU(s):")
    for i, gpu in enumerate(gpus):
        print(f"  - GPU[{i}]: {gpu}")
        tf.config.experimental.set_memory_growth(gpu, True) # Optional, but good practice
else:
    print("❌ FAILURE: TensorFlow did NOT find any GPUs.")
    print("   Please check your installation and CUDA/cuDNN setup.")
    
print("-" * 50)