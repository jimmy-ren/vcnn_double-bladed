# Shepard Convolutional Neural Networks
Before run the code, it's a good idea to read the instructions and see the videos on the VCNN website ([www.deeplearning.cc/vcnn](http://www.deeplearning.cc/vcnn)) and setup VCNN. The code runs in both Ubuntu 14.04 and Windows 7.

## Paper
Jimmy SJ. Ren, Li Xu, Qiong Yan, Wenxiu Sun, "[Shepard Convolutional Neural Networks](http://papers.nips.cc/paper/5774-shepard-convolutional-neural-networks.pdf)", Advances in Neural Information Processing Systems (NIPS 2015)

## Super-resolution
We provided the pre-trained models for 2x, 3x and 4x super-resolution. <br>
For PSNR comparison, following previous papers in the literature, we used single channel images. <br>
Please run this [script](https://github.com/jimmy-ren/vcnn_double-bladed/blob/master/applications/Shepard_CNN/Shepard_super_res/shepard_sr_x2_demo.m) to test 2x super-resolution. <br>
Please run this [script](https://github.com/jimmy-ren/vcnn_double-bladed/blob/master/applications/Shepard_CNN/Shepard_super_res/shepard_sr_x3_demo.m) to test 3x super-resolution. <br>
Please run this [script](https://github.com/jimmy-ren/vcnn_double-bladed/blob/master/applications/Shepard_CNN/Shepard_super_res/shepard_sr_x4_demo.m) to test 4x super-resolution. <br>
Most of the methods we compared against can be found in this [web page](http://www.vision.ee.ethz.ch/~timofter/ACCV2014_ID820_SUPPLEMENTARY/).
<br>

We also provided scripts to generate color images. <br>
Please run this [script](https://github.com/jimmy-ren/vcnn_double-bladed/blob/master/applications/Shepard_CNN/Shepard_super_res/color_super_res/color_shepard_sr_x2_demo.m) to test 2x color image super-resolution. <br>
Please run this [script](https://github.com/jimmy-ren/vcnn_double-bladed/blob/master/applications/Shepard_CNN/Shepard_super_res/color_super_res/color_shepard_sr_x3_demo.m) to test 3x color image super-resolution. <br>
Please run this [script](https://github.com/jimmy-ren/vcnn_double-bladed/blob/master/applications/Shepard_CNN/Shepard_super_res/color_super_res/color_shepard_sr_x4_demo.m) to test 4x color image super-resolution. <br>
<br>

To train a new model, run this [script](https://github.com/jimmy-ren/vcnn_double-bladed/blob/master/data/Shepard_CNN/Shepard_super_res/gen_training_data.m) to generate the data. You adjust the variable "down_factor" to indicate whether you would like to generate data for x2, x3 or x4 super-resolution. Also adjust the variable "training_mat_path" and "val_mat_path" to indicate the path to store the training and validation data. <br>

Once the data is ready, you may run this [script](https://github.com/jimmy-ren/vcnn_double-bladed/blob/master/applications/Shepard_CNN/Shepard_super_res/shepard_sr_x2_train.m) to train a x2 super-resolution model. This [script](https://github.com/jimmy-ren/vcnn_double-bladed/blob/master/applications/Shepard_CNN/Shepard_super_res/shepard_sr_x3_train.m) for x3 super-resolution model. This [script](https://github.com/jimmy-ren/vcnn_double-bladed/blob/master/applications/Shepard_CNN/Shepard_super_res/shepard_sr_x4_train.m) for x4 super-resolution model.

## Inpainting


