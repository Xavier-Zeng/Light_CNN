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
# from .cifar_squeezenet1_0_1_1_vanilla import squeezenet1_0, squeezenet1_1

# add cifar_squeezenext
# from .cifar_squeezenext import sqnxt_23_1x, sqnxt_23_1x_v5, sqnxt_23_2x, sqnxt_23_2x_v5

#add cifar_squeezenext_ibn_a_2 (在squeezenext block中的前两个1x1卷积核中都添加IBN)
from .cifar_squeezenext_ibn_a_2 import sqnxt_23_1x_ibn_a_2, sqnxt_23_1x_v5_ibn_a_2, sqnxt_23_2x_ibn_a_2, sqnxt_23_2x_v5_ibn_a_2

#add cifar_squeezenext_ibn_a_2_0_25 (在squeezenext block中的前两个1x1卷积核中都添加IBN，IN占总比例为0.25)
from .cifar_squeezenext_ibn_a_2_0_25 import sqnxt_23_1x_ibn_a_2_0_25, sqnxt_23_1x_v5_ibn_a_2_0_25, sqnxt_23_2x_ibn_a_2_0_25,  sqnxt_23_2x_v5_ibn_a_2_0_25

#add cifar_squeezenext_ibn_a_2_0_75 (在squeezenext block中的前两个1x1卷积核中都添加IBN，IN占总比例为0.75)
from .cifar_squeezenext_ibn_a_2_0_75 import sqnxt_23_1x_ibn_a_2_0_75, sqnxt_23_1x_v5_ibn_a_2_0_75, sqnxt_23_2x_ibn_a_2_0_75,  sqnxt_23_2x_v5_ibn_a_2_0_75

#add cifar_squeezenext_ibn_b
from .cifar_squeezenext_ibn_b import sqnxt_23_1x_ibn_b, sqnxt_23_1x_v5_ibn_b, sqnxt_23_2x_ibn_b, sqnxt_23_2x_v5_ibn_b

#add mobilenet for 224x224
from .mobilenet import mobilenet_0_25_224, mobilenet_0_5_224, mobilenet_0_75_224, mobilenet_1_0_224

#add mobilenet v2 for 224x224
from .mobilenet_v2 import mobilenet_v2_0_4x_224, mobilenet_v2_0_75x_224, mobilenet_v2_1x_224, mobilenet_v2_1_4x_224

#add squeezenext for 227x227
from .squeezenext import sqnxt_23_1x, sqnxt_23_1x_v5 ,sqnxt_23_2x, sqnxt_23_2x_v5
#add  squeezenext for 32x32
from .cifar_squeezenext import cifar_sqnxt_23_1x, cifar_sqnxt_23_1x_v5, cifar_sqnxt_23_2x, cifar_sqnxt_23_2x_v5   

#add  shufflenet for 224x224
from .shufflenet import shufflenet_g_1, shufflenet_g_2, shufflenet_g_3, shufflenet_g_4, shufflenet_g_8   

#add  shufflenet v2 for 224x224
from .shufflenet_v2 import shufflenet_v2_0_5x_224, shufflenet_v2_1x_224, shufflenet_v2_1_5x_224, shufflenet_v2_2x_224 

#add  shufflenet v2 for 32x32
from .cifar_shufflenet_v2 import cifar_shufflenet_v2_0_5x_32, cifar_shufflenet_v2_1x_32, cifar_shufflenet_v2_1_5x_32, cifar_shufflenet_v2_2x_32 

