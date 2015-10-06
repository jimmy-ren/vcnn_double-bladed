# VCNN - Double-Bladed Sword
Vectorized implementation of convolutional neural networks (CNN) in <b>Matlab</b> for both visual recognition and image processing. It's a unified framework for both high level and low level computer vision tasks.

## How to use it
You can <b>directly</b> try the demos without referring to any materials in the [project website](http://vcnn.deeplearning.cc). <br>
1. For MNIST, you can launch [this script](https://github.com/jimmy-ren/vcnn_double-bladed/blob/master/applications/MNIST/mnist_test_demo.m) to use a pre-trained model. For training, just launch [this script](https://github.com/jimmy-ren/vcnn_double-bladed/blob/master/applications/MNIST/mnist_train_demo.m). You will get sensible results within seconds.<br>
2. For image denoise, launch [this script](https://github.com/jimmy-ren/vcnn_double-bladed/blob/master/applications/image_denoise/denoise_test_demo.m) to see the denoise result by pre-train models. For training, you need to generate the data yourself since the data used in the training is large. Please do the following steps to generate data: a) download MIT saliency dataset from [here](http://saliency.mit.edu/BenchmarkIMAGES.zip) and put all the image files [here](https://github.com/jimmy-ren/vcnn_double-bladed/tree/master/data/denoise/mit_saliency); b) launch [this script](https://github.com/jimmy-ren/vcnn_double-bladed/blob/master/applications/image_denoise/gen_data/gen_training_data.m) to generate training data; c) launch [this script](https://github.com/jimmy-ren/vcnn_double-bladed/blob/master/applications/image_denoise/gen_data/gen_val_data.m) to generate validation data; d) launch [this script](https://github.com/jimmy-ren/vcnn_double-bladed/blob/master/applications/image_denoise/denoise_train_demo.m) to start the training.<br>

Please visit the [project website](http://vcnn.deeplearning.cc) for all documents, examples and videos.

## Hardware/software requirements
1. Matlab 2014b or later, CUDA 6.0 or later (currently tested in Ubuntu 14.04 and Windows 7)<br>
2. A Nvidia GPU with 2GB GPU memory or above (if you would like to run on GPU). You can also train a new model without a GPU by specifying "config.compute_device = 'CPU';" in the config file (e.g. [mnist_configure.m](https://github.com/jimmy-ren/vcnn_double-bladed/blob/master/applications/MNIST/mnist_configure.m)). <br>

## Videos
1. [Introduction](https://www.youtube.com/watch?v=aYhl_k51Tks)<br>
2. [MNIST example (demonstrate the speed & accuracy)](https://www.youtube.com/watch?v=6mMa59niBxo)<br>
3. [Image denoising example](https://www.youtube.com/watch?v=3Otm4sjhelg)<br>

## Contributors
[Jimmy SJ. Ren](http://www.jimmyren.com) (jimmy.sj.ren@gmail.com)<br>
[Li Xu](http://www.lxu.me) (nathan.xuli@gmail.com)

## Citation
Cite our papers if you find this software useful.<br>
1. Jimmy SJ. Ren and Li Xu, "[On Vectorization of Deep Convolutional Neural Networks for Vision Tasks](http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9988)", 
The 29th AAAI Conference on Artificial Intelligence (<b>AAAI-15</b>). Austin, Texas, USA, January 25-30, 2015<br>

## VCNN was used in the following research projects
1. Li Xu, Jimmy SJ. Ren, Ce Liu, Jiaya Jia, "[Deep Convolutional Neural Network for Image Deconvolution](http://papers.nips.cc/paper/5485-deep-convolutional-neural-network-for-image-deconvolution.pdf)", Advances in Neural Information Processing Systems (<b>NIPS 2014</b>).<br>
2. Li Xu, Jimmy SJ. Ren, Qiong Yan, Renjie Liao, Jiaya Jia, "[Deep Edge-Aware Filters](http://jmlr.org/proceedings/papers/v37/xub15.html)", The 32nd International Conference on Machine Learning (<b>ICML 2015</b>).<br>
3. Yongtao Hu, Jimmy SJ. Ren, Jingwen Dai, Chang Yuan, Li Xu, Wenping Wang, "[Deep Multimodal Speaker Naming](http://herohuyongtao.github.io/research/publications/speaker-naming/)", The 23rd ACM International Conference on Multimedia (<b>MM 2015</b>).<br>
4. Jimmy SJ. Ren, Li Xu, Qiong Yan, Wenxiu Sun, "Shepard Convolutional Neural Networks", Advances in Neural Information Processing Systems (<b>NIPS 2015</b>).<br>

