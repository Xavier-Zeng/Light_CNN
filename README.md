# SqueezeNet/SqueezeNext, MobileNet/v2, ShuffleNet/v2 Pytorch Implementation for CIFAR-10 and ImageNet
Architecture refer to [ResNext-DenseNet](https://github.com/D-X-Y/ResNeXt-DenseNet)
- SqueezeNet (SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size)
- SqueezeNext (SqueezeNext: Hardware-Aware Neural Network Design)
- MobileNet (Mobilenets: Efficient convolutional neural networks for mobile vision applications)
- MobileNet v2 (MobileNetV2: Inverted Residuals and Linear Bottlenecks)
- ShuffleNet (ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices)
- ShuffleNet v2 (Shufflenet v2: Practical guidelines for efficient cnn architecture design)

- [x] Train on CIFAR-10 and ImageNet(sample a small data) with SqueezeNet1.0, SqueezeNet1.1
- [x] Train on CIFAR-10 and ImageNet(sample a small data) with sqnxt_23_1x, sqnxt_23_1x_v5, sqnxt_23_2x, sqnxt_23_2x_v5

## Usage
To train on CIFAR-10 using 4 gpu:

```bash
python main.py ./data --dataset cifar10 --arch sqnxt_23_1x --save_path ./snapshots/cifar10_sqnxt_23_1x_300 --epochs 300 --learning_rate 0.1 --schedule 150 225 --gammas 0.1 0.1 --batch_size 128 --workers 4 --ngpu 1
```


## Configurations

### 1.  prepare data of CIFAR-10 and ImageNet(sample a small data) and store data in folder `data`.
```bash
data
|_ cifar-10-batches-py
|   |_ batches.meta
|   |_ data_batch_1
|   |_ data_batch_2
|   |_ data_batch_3
|   |_ data_batch_4
|   |_ data_batch_5
|   |_ readme.html
|   |_ test_batch

|_ ImageNet
|   |_ train
|      |_ classe1
|         |_ image1.jpg
|         |_ image2.jpg
|         |_ ...
|      |_ classe2
|      |_ ...
|      |_ classe3
|   |_ val
|      |_ classe1
|         |_ image1.jpg
|         |_ image2.jpg
|         |_ ...
|      |_ classe2
|      |_ ...
|      |_ classe3

```


## Other Projects
* [Torch (@facebookresearch)](https://github.com/facebookresearch/ResNeXt). (Original) CIFAR and ImageNet
* [MXNet (@dmlc)](https://github.com/dmlc/mxnet/tree/master/example/image-classification#imagenet-1k). ImageNet
* [PyTorch (@prlz77)](https://github.com/prlz77/ResNeXt.pytorch). CIFAR
* [EraseReLU](https://github.com/D-X-Y/EraseReLU). (will be public soon)

## Cite
```
@article{iandola2016squeezenet,
  title={Squeezenet: Alexnet-level accuracy with 50x fewer parameters and< 0.5 mb model size},
  author={Iandola, Forrest N and Han, Song and Moskewicz, Matthew W and Ashraf, Khalid and Dally, William J and Keutzer, Kurt},
  journal={arXiv preprint arXiv:1602.07360},
  year={2016}
}
@article{gholami2018squeezenext,
  title={SqueezeNext: Hardware-Aware Neural Network Design},
  author={Gholami, Amir and Kwon, Kiseok and Wu, Bichen and Tai, Zizheng and Yue, Xiangyu and Jin, Peter and Zhao, Sicheng and Keutzer, Kurt},
  journal={arXiv preprint arXiv:1803.10615},
  year={2018}
}
@article{howard2017mobilenets,
  title={Mobilenets: Efficient convolutional neural networks for mobile vision applications},
  author={Howard, Andrew G and Zhu, Menglong and Chen, Bo and Kalenichenko, Dmitry and Wang, Weijun and Weyand, Tobias and Andreetto, Marco and Adam, Hartwig},
  journal={arXiv preprint arXiv:1704.04861},
  year={2017}
}
@inproceedings{sandler2018mobilenetv2,
  title={MobileNetV2: Inverted Residuals and Linear Bottlenecks},
  author={Sandler, Mark and Howard, Andrew and Zhu, Menglong and Zhmoginov, Andrey and Chen, Liang-Chieh},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={4510--4520},
  year={2018}
}
@article{Zhang2017ShuffleNet,
  title={ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices},
  author={Zhang, Xiangyu and Zhou, Xinyu and Lin, Mengxiao and Sun, Jian},
  year={2017},
}
@article{ma2018shufflenet,
  title={Shufflenet v2: Practical guidelines for efficient cnn architecture design},
  author={Ma, Ningning and Zhang, Xiangyu and Zheng, Hai-Tao and Sun, Jian},
  journal={arXiv preprint arXiv:1807.11164},
  year={2018}
}
```
## Download the CIFAR-10 dataset

1. Download the images from http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

## Download the ImageNet dataset
The ImageNet Large Scale Visual Recognition Challenge (ILSVRC) dataset has 1000 categories and 1.2 million images. The images do not need to be preprocessed or packaged in any database, but the validation images need to be moved into appropriate subfolders.

1. Download the images from http://image-net.org/download-images

2. Extract the training data:
  ```bash
  mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
  tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
  find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
  cd ..
  ```

3. Extract the validation data and move images to subfolders:
  ```bash
  mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
  wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
  ```
