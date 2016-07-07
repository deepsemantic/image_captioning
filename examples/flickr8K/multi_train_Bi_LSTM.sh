#!/usr/bin/env bash

GPU_ID=0
CAFFE=/home/chengwang/workspace/chengwang_github/caffe_recurrent_dev
WEIGHTS=\
./models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel
$CAFFE/build/tools/caffe train \
    -solver ./examples/flickr8K/multi_Bi_LSTM_solver.prototxt \
    -weights $WEIGHTS \
    -gpu $GPU_ID
