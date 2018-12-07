"""The models subpackage contains definitions for the following model
architectures:
-  `SqueezeNext` for CIFAR10 CIFAR100
You can construct a model with random weights by calling its constructor:
.. code:: python
    import models
    sqnxt_23_1x = models.sqnxt_23_1x(num_classes)
    sqnxt_23_1x_v5 = models.sqnxt_23_1x_v5(num_classes)
    sqnxt_23_2x = models.sqnxt_23_2x(num_classes)
    sqnxt_23_2x_v5 = models.sqnxt_23_2x_v5(num_classes)


.. SqueezeNext: https://arxiv.org/abs/1803.10615
"""

#add cifar_squeezenet1_0_1_1_vanilla
from .cifar_squeezenet1_0_1_1_vanilla import squeezenet1_0, squeezenet1_1

# add cifar_squeezenext
from .cifar_squeezenext import sqnxt_23_1x, sqnxt_23_1x_v5, sqnxt_23_2x, sqnxt_23_2x_v5

#add cifar_squeezenext_ibn_a_2 (在squeezenext block中的前两个1x1卷积核中都添加IBN)
from .cifar_squeezenext_ibn_a_2 import sqnxt_23_1x_ibn_a_2, sqnxt_23_1x_v5_ibn_a_2, sqnxt_23_2x_ibn_a_2, sqnxt_23_2x_v5_ibn_a_2

#add cifar_squeezenext_ibn_a_2_0_25 (在squeezenext block中的前两个1x1卷积核中都添加IBN，IN占总比例为0.25)
from .cifar_squeezenext_ibn_a_2_0_25 import sqnxt_23_1x_ibn_a_2_0_25, sqnxt_23_1x_v5_ibn_a_2_0_25, sqnxt_23_2x_ibn_a_2_0_25,  sqnxt_23_2x_v5_ibn_a_2_0_25

#add cifar_squeezenext_ibn_a_2_0_75 (在squeezenext block中的前两个1x1卷积核中都添加IBN，IN占总比例为0.75)
from .cifar_squeezenext_ibn_a_2_0_75 import sqnxt_23_1x_ibn_a_2_0_75, sqnxt_23_1x_v5_ibn_a_2_0_75, sqnxt_23_2x_ibn_a_2_0_75,  sqnxt_23_2x_v5_ibn_a_2_0_75

#add cifar_squeezenext_ibn_b
from .cifar_squeezenext_ibn_b import sqnxt_23_1x_ibn_b, sqnxt_23_1x_v5_ibn_b, sqnxt_23_2x_ibn_b, sqnxt_23_2x_v5_ibn_b

#add mobilenet
from .mobilenet import mobilenet_224_1_0,