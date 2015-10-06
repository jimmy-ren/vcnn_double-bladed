# Deep Edge-Aware Filters
There are many edge-aware filters varying in their construction forms and filtering properties. We made the attempt to learn a big and important family of edge-aware operators from data. Our method gives rise to a powerful tool to approximate various filters without knowing the original models and implementation details. Fast approximation for complex edge-aware filters and achieves up to 200x acceleration.

## How to use it
Before run the code, it's a good idea to read the instructions and see the videos on the VCNN website ([www.deeplearning.cc/vcnn](http://www.deeplearning.cc/vcnn)) and setup VCNN.

<b>To run the pre-trained models</b>, run 'deepeaf_demo.m' in this application. There are currently 12 pre-trained filters. Please noted that Matlab 2014b + CUDA 6.0up + Nvidia GPU are needed to run the code. If you have a GPU with 4GB memory, you can directly run the demo. If you have a GPU with memory less than 4GB and encounter an out of memory error, you may go to "applications/deep_edge_aware_filters/utility/prepare_net_filter.m" to make the "max_patch_size" variable smaller.

<b>For training a new filter</b>, you may do as follows. Because we did not release our Flickr images, however, any natural image dataset shall do well (e.g. BSDS500 database). In that case, you need to firstly put the BSDS500 images under "data/deepeaf/BSDS500". Then run the "applications/deep_edge_aware_filters/utility/gen_training_data.m" script to generate the training data. The script will generate 101 files in the training sample folder (e.g. data/deepeaf/L0/training/), each file contains 10000 training samples. The first 100 files are for training. Put the last file into the validation folder (e.g. data/deepeaf/L0/val), load this file and change the variable from "samples" and "labels" to "test_samples" and "test_labels", save as "val_1.mat". Then you can run "deepeaf_training.m" to train a new model. You may want to look at aforementioned matlab scripts yourself, it is pretty straightforward.

## Citation
Cite our paper if you find this work useful.<br>
Li Xu, Jimmy SJ. Ren, Qiong Yan, Renjie Liao, Jiaya Jia, "[Deep Edge-Aware Filters](http://jmlr.org/proceedings/papers/v37/xub15.html)", The 32nd International Conference on Machine Learning (<b>ICML 2015</b>). Lille, France, July 6-11, 2015 <br>

## The learning system deep edge-aware filters run on
Jimmy SJ. Ren and Li Xu, "[On Vectorization of Deep Convolutional Neural Networks for Vision Tasks](http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9988)", 
The 29th AAAI Conference on Artificial Intelligence (<b>AAAI-15</b>). Austin, Texas, USA, January 25-30, 2015<br>


