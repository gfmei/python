#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on Tue Mar 08 15:07:19 2016

@author: gfmei
"""
from PIL import Image, ImageEnhance
from pylab import *
import rof

enhancer = ImageEnhance.Contrast(Image.open('ver0.jpg').convert("L"))
image_enhancer = enhancer.enhance(4)
im = array(image_enhancer)
bins = 256
imhist, bins = histogram(im.flatten(), bins)

cdf = imhist.cumsum()
cdf = 255 * cdf / cdf[-1]
im2 = interp(im.flatten(), bins[:-1], cdf)

U, T = rof.denoise(im2.reshape(im.shape), im2.reshape(im.shape))


arr = array(U)
gray()
imshow(arr)
for i in range(arr.shape[1]-1):
    if arr[0][i] < 250:
        arr[1][i] = 255
        arr[2][i] = 255
        arr[3][i] = 255
for i in range(arr.shape[0]-1):
    arr[i][arr.shape[1]-1] = 255
    arr[i-1][arr.shape[1]-1] = 255
    arr[i-2][arr.shape[1]-1] = 255
    arr[i][arr.shape[1]-2] = 255
    arr[i-1][arr.shape[1]-2] = 255
    arr[i-2][arr.shape[1]-2] = 255
for i in range(arr.shape[0]-6, arr.shape[0]-1):
    for j in range(arr.shape[1]-1):
        arr[i][j] = 255
for i in range(arr.shape[1]-5, arr.shape[1]-1):
    for j in range(arr.shape[0]-1):
        arr[j][i] = 255
for i in range(0, 5):
    for j in range(arr.shape[0]-1):
        arr[j][i] = 255
i = 4
j = 4
while i < array(U).shape[0]-6:
    while j < array(U).shape[1]-6:
        if arr[i][j] < 200 or arr[i][j-5] < 200 or arr[i][j+5] < 200 or arr[i+5][j] < 200 \
                or arr[i-5][j] < 200 or arr[i+5][j+5] < 200 or arr[i-5][j-5] < 200 or arr[i-5][j+5] < 200:
            for k in range(i-6, i+6):
                for l in range(j-6, j+6):
                    arr[k][l] = 255
        j += 1
    i += 1
w = array(U).shape[0]-5
h = array(U).shape[1]-5
while w > 4:
    while h > 4:
        if (arr[i][j] < 200) or (arr[i-5][j-5] < 200) or (arr[i][j-5] < 200) or (arr[i+5][j-5] < 200) \
                or arr[i-5][j] < 200 or arr[i+5][j] < 200 or arr[i-5][j+5] < 200 or arr[i+5][j+5] < 200\
                or arr[i-3][j-3] < 200 or arr[i][j-3] < 200 or arr[i+3][j-3] < 200 \
                or arr[i-3][j] < 200 or arr[i+3][j] < 200 or arr[i-3][j+3] < 200 or arr[i+3][j+3] < 200:
            for k in range(i-6, i+6):
                for l in range(j-6, j+6):
                    arr[k][l] = 255
        h -= 1
    w -= 1
while i < array(U).shape[0]-3:
    while j < array(U).shape[1]-3:
        if arr[i][j] < 200 or arr[i][j-3] < 200 or arr[i][j+3] < 200 or arr[i+3][j] < 200 or arr[i-3][j] < 200:
            for k in range(i-3, i+3):
                for l in range(j-3, j+3):
                    arr[k][l] = 255
        j += 1
    i += 1

pil_im = Image.fromarray(arr)
data = pil_im.getdata()
w, h = pil_im.size
black_point = 0
for x in xrange(1, w-1):
    for y in xrange(1, h-1):
        mid_pixel = data[w*y+x]  # 中央像素点像素值
        if mid_pixel < 150:  # 找出上下左右四个方向像素点像素值
            pil_im.putpixel((x, y), 255)
            pil_im.putpixel((x-1, y-1), 255)
            pil_im.putpixel((x, y-1), 255)
            pil_im.putpixel((x+1, y-1), 255)
            pil_im.putpixel((x-1, y), 255)
            pil_im.putpixel((x+1, y), 255)
            pil_im.putpixel((x-1, y+1), 255)
            pil_im.putpixel((x, y+1), 255)
            pil_im.putpixel((x+1, y+1), 255)


pil_im = array(arr - array(pil_im))
pil_im = Image.fromarray(uint8(pil_im))
data = pil_im.getdata()
w, h = pil_im.size
for x in xrange(1, w-1):
    for y in xrange(1, h-1):
        mid_pixel = data[w*y+x]  # 中央像素点像素值
        if mid_pixel == 0:  # 找出上下左右四个方向像素点像素值
            pil_im.putpixel((x, y), 255)
            pil_im.putpixel((x, y-1), 255)
            pil_im.putpixel((x-1, y), 255)
            pil_im.putpixel((x+1, y), 255)
            pil_im.putpixel((x, y+1), 255)
figure()
imshow(pil_im)
show()