#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on Tue Mar 08 15:07:19 2016

@author: gfmei
"""
from pylab import *
import MySQLdb
import iden_apply
# process images at fixed size (50,50)
try:
    conn = MySQLdb.connect(host='localhost', user='root', passwd='51desk', db='Place51desk', charset='utf8', port=3306)
    cur = conn.cursor()
    label = iden_apply.ver_result()
    cur.execute("select content from ver_image where label = %s" % label)
    results = cur.fetchall()
    cur.close()
    conn.close()
except Exception, e:
    print e.message
print results[0][0]