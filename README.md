# Image Captioning with Deep Bidirectional LSTMs

This branch hosts the code for our paper accepted at ACMMM 2016 ["Image Captioning with Deep Bidirectional LSTMs"](http://arxiv.org/abs/1604.00790), to see [Demonstration](https://youtu.be/a0bh9_2LE24)

### Features 
 - Training with Bidirectional LSTMs
 - Implemented data augmentation: multi-crops, multi-scale, vectical mirroring
 - Variant Bidirectional LSTMs: Bi-F-LSTM, Bi-S-LSTM

### Usage and Example 
 - This work extends ["Long-term Recurrent Convolutional Networks (LRCN)"](http://jeffdonahue.com/lrcn/) to bidirectional LSTMs with data augmentation
 - We provide an example [flickr8K](http://nlp.cs.illinois.edu/HockenmaierGroup/8k-pictures.html), in which you can train proposed networks
 - (1) download flickr8 training and test images, and put it to "data/flickr8K/images/", the dataset splits can be found in "data/flickr8K/texts/"
 - (2) create databases with "flickr8K_to_hdf5_data_forward.py" and "flickr8K_to_hdf5_data_backward.py" 
 - (3) train network with "multi_train_Bi_LSTM.sh"
 - (4) perform image caption generation and image-sentence retrieval experiments with "bi_generation_retrieval.py" 
 
### Citation

Please cite in your publications if it helps your research:

    @article{wang2016image,
    title={Image Captioning with Deep Bidirectional LSTMs},
    author={Wang, Cheng and Yang, Haojin and Bartz, Christian and Meinel, Christoph},
    journal={arXiv preprint arXiv:1604.00790},
    url={http://arxiv.org/abs/1604.00790}
    year={2016}
    }
----
Following is orginal README of Caffe
# Caffe

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BVLC reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
