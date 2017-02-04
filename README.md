# Residual Networks of Residual Networks in Keras 

This is an implementation of the paper ["Residual Networks of Residual Networks: Multilevel Residual Networks"](https://arxiv.org/pdf/1608.02908v1.pdf)

# Explanation

Ordinarily, Residual networks have hundreds or even thousands of layers to accurately classify images in major image recognition tasks, but
building a network by simply stacking residual blocks inevitably limits its optimization ability.

This paper attempts to improve the optimization ability of Residual Networks by adding level-wise shortcut connections upon original residual networks, to promote the learning capability of residual networks. 

This can be shown by the figure from the paper : <br>
<img src="https://github.com/titu1994/Residual-of-Residual-Networks/blob/master/images/resnet%20vs%20ror.JPG?raw=true" width=50%>

There are two different architectures available, since RoR can be extended to Wide Residual Networks or Pre-ResNets as well.

The two images below from the paper describe the architecture of RoR on ResNets and RoR on Wide Residial Networks: <br>
<img src="https://github.com/titu1994/Residual-of-Residual-Networks/blob/master/images/ror-3.JPG?raw=true" width=49% height=600> <img src="https://github.com/titu1994/Residual-of-Residual-Networks/blob/master/images/ror-wrn.JPG?raw=true" width=50% height=600>

The classification accuracy of these networks on CIFAR 10 (from the paper) are : <br>
<img src="https://github.com/titu1994/Residual-of-Residual-Networks/blob/master/images/ror-accuracy-cifar10.JPG?raw=true">

# Usage

The paper uses several models such as RoR-3-110 (ResNet-101) and and RoR-3-WRN-40-2 (Wide Residual Network-40-2) models but due to GPU memory limitations, only the weights for the RoR-3-WRN-40-2 have been provided in the [Releases tab](https://github.com/titu1994/Residual-of-Residual-Networks/releases)

Please download the weights and place them in the weights folder.

To create RoR ResNet models, use the `ror.py` script :
```
import ror

input_dim = (3, 32, 32) if K.image_dim_ordering() == 'th' else (32, 32, 3)
model = ror.create_residual_of_residual(input_dim, nb_classes=100, N=2, dropout=0.0) # creates RoR-3-110 (ResNet)
```

To create RoR Wide Residual Network models, use the `ror_wrn.py` script : 
```
import ror_wrn as ror

input_dim = (3, 32, 32) if K.image_dim_ordering() == 'th' else (32, 32, 3)
model = ror.create_pre_residual_of_residual(input_dim, nb_classes=100, N=6, k=2, dropout=0.0) # creates RoR-3-WRN-40-2 (WRN)
```

# Performance

The RoR-WRN-40-2 model described in the paper requires 500 epochs to acheive a classification accuracy of 94.99 % (5.01 % error).

The Theano weights provided for this model are trained for 100 epochs using Adam with a learning rate of 1e-3, which achieves a classification accuracy of 94.48% (5.52 % error)

# Requirements

- Keras 
- Theano (weights provided) / Tensorflow (weights not yet converted)
- scipy
- h5py
- sklearn (for metrics)
