# Grouping Similar Images Using Pretrained CNN
Here I have used pretrained [**MobileNetV2**](https://keras.io/api/applications/mobilenet/#mobilenetv2-function).

Since the model is pretrained on ImageNet dataset the outputs without the softmax will be a vector of shape 1000. Similar images should have similar output vectors from the model. So we will group the similar images by
calculating the cosine angle between the vectors. If the images are similar then the cosine of angle between the vectors should be close to 1.
