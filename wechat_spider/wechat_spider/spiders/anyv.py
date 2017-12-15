#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: gfmei
finish date: 20171214
"""

import scrapy
import re
import requests
import MySQLdb
import traceback

def get_address_urls():
    r = requests.get("http://anyv.net/")
    data = r.text
    address_urls = re.findall(r"http://www.anyv.net/index.php/category-\d\d?\d?", data)
    address_urls = list(set(address_urls))
    return address_urls

def get_start_urls():
    start_urls = []
    address_urls = get_address_urls()
    for s in address_urls:
        print(s)
        r1 = requests.get(s)
        data1 = r1.text
        temp = re.findall(r"http://www.anyv.net/index.php/category-\d\d?\d?-page-(\d?\d?)", data1)
        start_urls.append(s)
        if len(temp):
            page_num = max([int(x) for x in temp])
            for i in range(1, page_num + 1):
                start_urls.append(s + '-page-' + str(i))
    start_urls = list(set(start_urls))
    return start_urls


class AnyvSpider(scrapy.spiders.Spider):
    # 爬虫解析
    name = 'anyv'
    allowed_domains = ['anyv.net']
    start_urls = get_start_urls()
    index2name = {1: u'新闻', 2: u'财经', 3: u'科技', 4: u'情感', 5: u'阅读', 6: u'搞笑',7: u'趣玩', 8: u'时尚', 9: u'生活',
                  10: u'健康', 11: u'旅游', 12: u'运动', 13: u'影音', 14: u'教育', 15: u'品牌', 16: u'购物', 17: u'明星',
                  18: u'名人', 19: u'游戏', 20: u'美女', 21: u'其它'}
    name2index = dict((v, k) for k, v in index2name.iteritems())


    def parse(self, response):
        label_name = response.xpath("/html/head/title//text()").extract()[0].split(' - ')[0][0:2]
        names = response.xpath("//div/ul[@class='clearfix']/div[@class='newpicsmall_list']"
                               "/a/li[@class='xiaobiaotizi']//text()").extract()
        try:
            conn = MySQLdb.connect(host='localhost', user='root', passwd='', db='robot', charset='utf8', port=3306)
            cur = conn.cursor()
            for name in names:
                value = [0, name, self.name2index[label_name]]
                cur.execute("insert into wx_materials(type, name, label) values ('%d', '%s', '%d')" % tuple(value))
            conn.commit()
            cur.close()
            conn.close()
        except Exception, e:
            print e.message
            traceback.print_exc()
        print label_name, str(names).decode('unicode-escape')
