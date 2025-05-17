# cnn-hyperparameter-optimization-pathmnist
This project explores the classification of medical images using a custom-built Convolutional Neural Network (CNN) on the PathMNIST dataset, part of the MedMNIST collection. The core focus is on hyperparameter testing to identify the optimal CNN architecture for multi-class classification.

## 🔍Highlights

- Dataset: PathMNIST (Medical histopathology images)
- Framework: PyTorch
- CNN with configurable layers, dropout, activations, pooling, and fully connected units
- Systematic testing for:
  - Kernel size
  - Dropout rates (conv & fc)
  - Activation functions (`ReLU`, `Tanh`, `Sigmoid`)
  - Optimizer type (`Adam`, `SGD`, `RMSprop`)
  - Batch size and epoch count

##  Objective

To evaluate the impact of various hyperparameters on the performance of a medical image classification model and determine the best architecture for generalization.

##  Best Parameters (as of final test)

```python
params = {
    'conv_kernel_size': 3,
    'conv_dropout': 0.4,
    'activation_fn': 'relu',
    'batch_size': 64,
    'optimizer_type': 'adam',
    'num_conv_layers': 2,
    'fc_hidden_units': 128,
    'pool_kernel_size': 3,
    'pool_stride': 2,
    'fc_dropout': 0.2,
    'num_epochs': 15,
    'num_classes': 9
}
