#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: gfmei
finish date: 20171214
"""

from lxml import html
import re
import requests
import traceback
from wxdb import WXDB

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

def get_address_urls():
    try:
        r = requests.get("http://anyv.net/")
        data = r.text
        address_urls = re.findall(r"http://www.anyv.net/index.php/category-\d\d?\d?", data)
    except Exception as e:
        print(e.message)
    address_urls = list(set(address_urls))
    return address_urls

def get_start_urls():
    start_urls = []
    address_urls = get_address_urls()
    for s in address_urls:
        print(s)
        r = requests.get(s)
        data = r.text
        try:
            temp = re.findall(r"http://www.anyv.net/index.php/category-\d\d?\d?-page-(\d?\d?)", data)
        except Exception as e:
            print(e.message)
        start_urls.append(s)
        if len(temp):
            page_num = max([int(x) for x in temp])
            for i in range(1, page_num + 1):
                start_urls.append(s + '-page-' + str(i))
    start_urls = list(set(start_urls))
    return start_urls


class AnyvSpider(object):
    # 爬虫解析
    start_urls = get_start_urls()
    index2name = {1: u'新闻', 2: u'财经', 3: u'科技', 4: u'情感', 5: u'阅读', 6: u'搞笑', 7: u'趣玩', 8: u'时尚', 9: u'生活',
                  10: u'健康', 11: u'旅游', 12: u'运动', 13: u'影音', 14: u'教育', 15: u'品牌', 16: u'购物', 17: u'明星',
                  18: u'名人', 19: u'游戏', 20: u'美女', 21: u'其它'}
    name2index = dict((v, k) for k, v in index2name.iteritems())

    db = WXDB()

    for start_url in start_urls:
        req = requests.get(start_url)
        tree = html.fromstring(req.text)

        label_name = tree.xpath("/html/head/title//text()")[0].split('-')[0][0:2]
        names = tree.xpath("//div/ul[@class='clearfix']/div[@class='newpicsmall_list']"
                           "/a/li[@class='xiaobiaotizi']//text()")
        print label_name, str(names).decode('unicode-escape')
        for name in names:
            value = [0, name, name2index[label_name]]
            db.execute("insert into wx_materials(type, name, label) values ('%d', '%s', '%d')" % tuple(value))

    # game
    req = requests.get('http://game.anyv.net/')
    tree = html.fromstring(req.text)
    print(tree)
    names = tree.xpath("//div[@class='section group']//h4/a//text()")
    print(str(names).decode('unicode-escape'))
    try:
        for name in names:
            value = [0, name, name2index[u'游戏']]
            db.execute("insert into wx_materials(type, name, label) values ('%d', '%s', '%d')" % tuple(value))
    except Exception, e:
        print e.message
        traceback.print_exc()

if __name__ == '__main__':
    anv = AnyvSpider()