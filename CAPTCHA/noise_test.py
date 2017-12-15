#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on Wed Mar 09 15:07:19 2016

@author: gfmei

"""
from PIL import Image, ImageDraw, ImageFont
import MySQLdb
from pylab import *
import rem_noise
im = rem_noise.get_image('captcha12.jpg')
im.show()