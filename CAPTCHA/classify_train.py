#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on Mar 22 15:45:19 2016

@author: gfmei
"""
import MySQLdb
import rem_noise
from pylab import *
try:
    conn = MySQLdb.connect(host='localhost', user='root', passwd='51desk', db='Place51desk', charset='utf8', port=3306)
    cur = conn.cursor()
    cur1 = conn.cursor()
    cur.execute("select id, label from ver_image where label = 0")
    cur1.execute("select id, label from ver_image where label = 1")
    results = cur.fetchall()
    results1 = cur1.fetchall()
    cur.close()
    cur1.close()
    conn.close()
except Exception, e:
    print e.message
class1 = []
class2 = []
labels = []
for row in results:
    pic = row[0] + '.jpg'
    im = array(rem_noise.get_image(pic))
    class1.append(im.reshape(-1, 1))
    labels.append(0)
for row in results1:
    pic = row[0] + '.jpg'
    im = array(rem_noise.get_image(pic))
    class2.append(im.reshape(-1, 1))
    labels.append(1)

print type(class1)
# bc = bayes.BayesClassifier()
# bc.train([array(class1).reshape(array(class1).shape[0], -1), array(class2).reshape(array(class2).shape[0], -1)], [-1, 1])
# im = rem_noise.get_image(pic)
# print bc.classify(array(class2).reshape(array(class2).shape[0], -1)[:2])[0]