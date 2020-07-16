# Tensor-Turbo
####CPU based ultra-fast Image Classifier based on TF 2 and TF Hub for Transfer Learning

## Project Overview
TensorTurbo is an Image Classification project aimed at training large scale datasets on CPU systems. This is a more efficient implementation of transfer learning. The high level steps of implementation include :
1. Compute bottleneck values - resultant feature vectors from a forward pass of an image
2. Create a TFRecord file based on bottleneck vectors and the groundtruths
3. Use multi-processing to parallelize data-preprocessing (sub-1 second to process over 10,000 images)
4. Train using tf.GradientTape and tf.data objects 

### Installation

This project works on TensorFlow 2 only. Clone this repository
```
https://github.com/Ashwin-Ramesh2607/Tensor-Turbo.git
cd Tensor-Turbo
```

Install all packages and dependencies 
```
pip3 install -r requirements.txt
```

### Benchmarking

The drastic speed-up can be noticed even when shifting the training to CPU.

method|latency per epoch
-|-
standard tf.keras|150 seconds (Google Colab **GPU**)
Tensor Turbo|1.1 seconds (Google Colab **CPU**)
