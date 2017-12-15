#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on Mar 22 15:45:19 2016

@author: gfmei
"""
import MySQLdb
import rem_noise
from svmutil import *
from pylab import *
try:
    conn = MySQLdb.connect(host='localhost', user='root', passwd='51desk', db='Place51desk', charset='utf8', port=3306)
    cur = conn.cursor()
    cur.execute("select id, label from ver_image where label = 0")
    results = cur.fetchall()
    cur1 = conn.cursor()
    cur1.execute("select id, label from ver_image where label = 1")
    results1 = cur1.fetchall()
    cur2 = conn.cursor()
    cur2.execute("select id, label from ver_image where label = 2")
    results2 = cur2.fetchall()
    conn.commit()
    cur.close()
    cur1.close()
    cur2.close()
    conn.close()
except Exception, e:
    print e.message
class1 = []
class2 = []
labels = []
test = []
t_labels = []

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
# samples = map(list, array(class1).reshape(array(class1).shape[0], -1))+\
#           map(list, array(class2).reshape(array(class2).shape[0], -1))
for row in results2:
    pic = row[0] + '.jpg'
    im = array(rem_noise.get_image(pic))
    test.append(im.reshape(-1, 1))
    labels.append(2)
    t_labels.append(2)
samples = map(list, array(class1).reshape(array(class1).shape[0], -1)) + \
          map(list, array(class2).reshape(array(class2).shape[0], -1)) + \
          map(list, array(test).reshape(array(test).shape[0], -1))
ex = array(rem_noise.get_image('captcha.jpg')).reshape(1, -1)
prob = svm_problem(labels, samples)
param = svm_parameter('-t 2')
m = svm_train(prob, param)
p_label, p_acc, p_val = svm_predict([2], map(list, array(ex).reshape(array(ex).shape[0], -1)), m)
print p_label