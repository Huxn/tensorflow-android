# 此项目分为tensorflow与android两部分

## tensorflow用于训练数据集，生成模型

* 使用python IDE打开
* 将mnist数据集转换成一张张图片并记住路径，我这里是项目路径下的mnist_digits_images中
* 修改代码中对应的图片路径
* 生成的模型保存在项目路径下的pic_train/mnist.pb

## android使用tensorflow生成的模型去识别手写0-9

* 将libandroid_tensorflow_inference_java.jar 与 armeabi/libtensorflow_inference.so 文件放在libs路径下
* 将tensorflow生成的模型mnist.pb保存在assets中
* 将要识别的手写图片放在drawable/image.jpg中
* 在app项目的gradle文件中添加implementation files('libs/libandroid_tensorflow_inference_java.jar')依赖