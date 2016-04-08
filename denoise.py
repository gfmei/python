#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on Tue Mar 08 15:07:19 2016

@author: gfmei
"""

from PIL import Image, ImageFilter, ImageDraw, ImageFont, ImageEnhance


# 打开一个jpg图像文件
im = Image.open('ver0.jpg')
#二值数组
t2val = {}
def twoValue(image, G):
    for y in xrange(0, image.size[1]):
        for x in xrange(0, image.size[0]):
            g = image.getpixel((x, y))
            if g > G:
                t2val[(x, y)] = 1
            else:
                t2val[(x, y)] = 0
enhancer = ImageEnhance.Contrast(im)
image_enhancer = enhancer.enhance(4)
img = im.convert("RGBA")
pix_data = img.load()

for y in xrange(img.size[1]):
    for x in xrange(img.size[0]):
        if pix_data[x, y][0] < 189:
            pix_data[x, y] = (0, 0, 0, 255)

for y in xrange(img.size[1]):
    for x in xrange(img.size[0]):
        if pix_data[x, y][1] < 200:
            pix_data[x, y] = (0, 0, 0, 255)

for y in xrange(img.size[1]):
    for x in xrange(img.size[0]):
        if pix_data[x, y][2] > 0:
            pix_data[x, y] = (255, 255, 255, 255)


def clearNoise(image, N, Z):

    for i in xrange(0, Z):
        t2val[(0, 0)] = 1
        t2val[(image.size[0] - 1, image.size[1] - 1)] = 1

        for x in xrange(1,image.size[0] - 1):
            for y in xrange(1,image.size[1] - 1):
                nearDots = 0
                L = t2val[(x,y)]
                if L == t2val[(x - 1,y - 1)]:
                    nearDots += 1
                if L == t2val[(x - 1,y)]:
                    nearDots += 1
                if L == t2val[(x- 1,y + 1)]:
                    nearDots += 1
                if L == t2val[(x,y - 1)]:
                    nearDots += 1
                if L == t2val[(x,y + 1)]:
                    nearDots += 1
                if L == t2val[(x + 1,y - 1)]:
                    nearDots += 1
                if L == t2val[(x + 1,y)]:
                    nearDots += 1
                if L == t2val[(x + 1,y + 1)]:
                    nearDots += 1
                if nearDots < N:
                    t2val[(x,y)] = 1
img = img.convert("L")
#图像显示
img.show()
# 获得图像尺寸:
w, h = im.size
# 缩放到50%:
# im.thumbnail((w//2, h//2))
# 图片的模糊化
#im2 = im.filter(ImageFilter.BLUR)
# 把模糊后的图像用jpeg格式保存:
#im2.save('E:/PycharmProjects/image/captcha_blur.jpg', 'jpeg')


# 模糊:
# image = image.filter(ImageFilter.BLUR)
# image.save('code.jpg', 'jpeg')

# 图片存储格式 大小 色彩模式
print im.format, im.size, im.mode