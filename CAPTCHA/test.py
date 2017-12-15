#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on Wed Mar 09 15:07:19 2016

@author: gfmei

"""
from PIL import Image, ImageDraw, ImageFont
import MySQLdb
from pylab import *

import random

#随机颜色1:
def rndColor():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

# 180 x 60:
width = 60 * 4
height = 100
expand = 20
try:
    conn = MySQLdb.connect(host='localhost', user='root', passwd='51desk', db='Place51desk', charset='utf8', port=3306)
    cur = conn.cursor()
    for num in range(504, 505):
        for k in range(0, 100):
            # 格式： Image.new(mode, size, color) => image
            image = Image.new('RGB', (width, height), (225, 255, 255))
            replica = Image.new('RGB', (width + expand, 80), (225, 255, 254))
            # 创建Font对象:
            font = ImageFont.truetype('C:\Windows\Fonts\simsun.ttc', 40)
            # 创建Draw对象:
            draw = ImageDraw.Draw(image)

            # 输出文字:
            word = u'贼眉鼠眼'
            for t in range(4):
                draw.text((60 * t + 10, 25), word[t], font=font, fill=rndColor())
            f_out = open('idiom_an.txt', 'a')
            f_out.writelines("ver" + str(k) + ":" + word.encode("utf-8") + "\n")
            f_out.close()
            # 直接复制图像
            # box = image.copy()
            box1 = (10, 10, 60, 65)
            box2 = (62, 10, 120, 65)
            box3 = (122, 10, 180, 65)
            box4 = (182, 10, 240, 65)
            # 图片的裁剪域旋转
            region1 = image.rotate(random.randint(-10, 5))
            region2 = image.rotate(random.randint(-10, 10))
            region3 = image.rotate(random.randint(-10, 10))
            region4 = image.rotate(random.randint(-5, 10))
            region1 = region1.crop(box1)
            region2 = region2.crop(box2)
            region3 = region3.crop(box3)
            region4 = region4.crop(box4)
            # 图片的粘贴
            step = 3
            box10 = (10 + expand, 10, 60 + expand, 65)
            box20 = (42 + 2 * step + expand, 10, 100 + 2 * step + expand, 65)
            box30 = (92 + 3 * step + expand, 10, 150 + 3 * step + expand, 65)
            box40 = (132 + 4 * step + expand, 10, 190 + 4 * step + expand, 65)
            replica.paste(region1, box10)
            replica.paste(region2, box20)
            replica.paste(region3, box30)
            replica.paste(region4, box40)
            # 填充每个像素:
            count = 600
            cou = 20
            for x in range(count):
                ImageDraw.Draw(replica).point((random.randint(0, width), random.randint(0, height)), fill=rndColor())
            i = 0
            for i in range(cou):
                ImageDraw.Draw(replica).line(((random.randint(0, width), random.randint(0, height)),
                                              (random.randint(0, width), random.randint(0, height))), rndColor())
            box = (25, 8, 214, 75)
            outfile = "E:/PycharmProjects/ver_image/image/"+"ver" + str(num) + "_" + str(k) + ".jpg"
            # 图片存储格式 大小 色彩模式
            replica.crop(box).save(outfile, 'jpeg')
            value = ["ver" + str(num) + '_' + str(k), word, num]
            cur.execute("insert into ver_image(id, content, label)values (%s, %s, %s)", tuple(value))
            conn.commit()
    cur.close()
    conn.close()
except Exception, e:
    print e.message