#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on Tue Mar 08 15:07:19 2016

@author: gfmei
"""
from pylab import *
import os, python.sift
def read_gesture_features_labels(path):
    # create list of all files ending in .dsift
    featlist = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.dsift')]
    # read the features
    features = []
    for featfile in featlist:
        l, d = python.sift.read_features_from_file(featfile)
    features.append(d.flatten())
    features = array(features)
    # create labels
    labels = [featfile.split('/')[-1][0]
    for featfile in featlist]
    return features, array(labels)