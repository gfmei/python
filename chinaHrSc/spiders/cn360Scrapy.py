#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: gfmei
finish date: 20160406
"""

import scrapy
import re,os
import requests
import MySQLdb
import traceback
from scrapy.selector import Selector

def get_address_urls():
    address_urls = []
    r = requests.get("http://www.cn360cn.com/")
    data = r.text
    temp = re.findall(r"province_\d+?\.aspx", data)
    address_urls += temp
    address_urls = list(set(address_urls))
    for i in range(len(address_urls)):
        address_urls[i] = "http://www.cn360cn.com/" + address_urls[i]
    return address_urls

def get_start_urls():
    start_urls = []
    temp = []
    address_urls = get_address_urls()
    for s in address_urls:
        r1 = requests.get(s)
        data1 = r1.text
        temp0 = re.findall(r"\w*?/\d*?/index\.htm", data1)
        temp += temp0
    temp = list(set(temp))
    for i in range(len(temp)):
        temp[i] = "http://www.cn360cn.com/" + temp[i]

    for url in temp[40:50]:
        ie = 1
        if_continue = True
        while if_continue:
            print url
            r = requests.get(url)
            if r.status_code == 200:

                data = r.text
                link_list = re.findall(r"\w*?\d*?\.htm", data)
                for i in range(len(link_list)):
                    link_list[i] = os.path.dirname(url) + '/' + link_list[i]
                start_urls += link_list
                ie += 1
                url = "/".join(url.split("/")[:-1] + ["index_%d.htm" % ie])
            else:
               if_continue = False
        # break
    start_urls = list(set(start_urls))
    return start_urls

class Cn360Spider(scrapy.spiders.Spider):

    name = 'cn360cn'
    allowed_domains = ['cn360cn.com']
    start_urls = get_start_urls()

    def parse(self, response):
        name = response.xpath("//head//title//text()").extract()[0].split(u',')[0].encode('utf-8')
        tel = response.xpath("//head//title//text()").extract()[0].split(u',')[1].encode('utf-8')
        address = response.xpath("//div[@class = 'cleft']/table/tr[4]/td//text()").extract()[1].encode('utf-8').\
            replace("\r", "").replace("\n", "").strip()
        province = response.xpath("//div[@class = 'cleft']/table/tr[4]/td//text()").extract()[1].encode('utf-8').split()[0]
        district = response.xpath("//div[@class = 'cleft']/table/tr[4]/td//text()").extract()[1].encode('utf-8').split()[1]
        description = response.xpath("//div[@class = 'divcontent']//text()").extract()[0].encode('utf-8').strip()
        try:
            conn = MySQLdb.connect(host='localhost', user='root', passwd='51desk', db='Place51desk', charset='utf8', port=3306)
            cur = conn.cursor()
            value = [name, tel, name, '中国', province, address, description, 156, 'http://www.cn360cn.com/', 20160407, 20160407, district]
            cur.execute("insert into company(name, tel, baidu_name, country, province, city, description, country_num,\
            source, cre_dt, upd_dt, district) values ('%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s')" % tuple(value))
            conn.commit()
            cur.close()
            conn.close()
        except Exception, e:
            print e.message
            traceback.print_exc()
        print name,tel
        print "*****************************************************"
