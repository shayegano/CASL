#!/bin/bash

sudo pip install python_speech_features seaborn numpy scipy matplotlib pandas sympy nose opencv-python

# Dependency needed to fix python double free/malloc corrupt error when calling with tensorflow
# See: https://github.com/tensorflow/tensorflow/issues/6968
sudo apt-get install libtcmalloc-minimal4
echo "export LD_PRELOAD='/usr/lib/libtcmalloc_minimal.so.4'" >> ~/.bashrc
source ~/.bashrc 
