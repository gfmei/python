# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# # author: gfmei
# # finish date: 20160302
# import scrapy
# import re
# import requests
# import MySQLdb
# import traceback
# from scrapy.selector import Selector
#
# def get_address_urls():
#     address_urls = []
#     r = requests.get("http://www.chinahr.com/company/")
#     data = r.text
#     temp = re.findall(r"(http://www.chinahr.com/company/(?![\di]).+?/)\"", data)
#     address_urls += temp
#     address_urls = list(set(address_urls))
#     return address_urls
#
#
# def get_start_urls():
#     start_urls = []
#     temp = []
#     address_urls = get_address_urls()
#     for s in address_urls:
#         r1 = requests.get(s)
#         data1 = r1.text
#         temp0 = re.findall(r"http://www.chinahr.com/company/.+?/p\d*/", data1)
#         temp += temp0
#     temp = list(set(temp))
#
#     for url in temp:
#         r = requests.get(url)
#         data = r.text
#         link_list = re.findall(r"http://www.chinahr.com/company/.+?html", data)
#         start_urls += link_list
#     start_urls = list(set(start_urls))
#     print len(start_urls), "CCCCCCCCCC"
#     return start_urls
#
# class ChinahrSpider(scrapy.spiders.Spider):
#
#     name = 'chinahr'
#     allowed_domains = ['chinahr.com']
#     start_urls = get_start_urls()
#
#     def parse(self, response):
#         name = response.xpath("//div[@class = 'main']//h1//text()").extract()[0]
#         address = response.xpath("//div[@class = 'wrap-mc']/em/text()").extract()[0] + ',' +\
#                   response.xpath("//div[@class = 'address']//i[@class = 'icon_hf add']/parent::p//text()").\
#                   extract()[0].replace(u"公司地址：", "").strip()
#         province = response.xpath("//div[@class = 'wrap-mc']/em/text()").extract()[0]
#         category = response.xpath("//div[@class = 'wrap-mc']/em/text()").extract()[1].strip()
#         proper = response.xpath("//div[@class = 'wrap-mc']/em/text()").extract()[2].strip()
#         scale = ''
#         if u"注册资金" in response.xpath("//div[@class = 'wrap-mc']/em/text()").extract()[3].strip():
#             scale = response.xpath("//div[@class = 'wrap-mc']/em/text()").extract()[3].strip()
#         description = response.xpath("//div[@class = 'intro-company']//div[@class = 'art-company']//div[@class = 'article']//text()").extract()
#         description = [p.replace("\r", "").replace("\n", "").strip() for p in description if p.replace("\r", "").replace("\n", "").strip()]
#         try:
#             conn = MySQLdb.connect(host='localhost', user='root', passwd='51desk', db='Place51desk', charset='utf8', port=3306)
#             cur = conn.cursor()
#             value = [name, '', proper, name, category, '中国', province, address, description, 156, 'http://www.chinahr.com//',20130307, 20130307]
#             cur.execute("insert into company(name, scale, property, baidu_name, category, country, province, city, description,\
#             country_num, source, cre_dt, upd_dt) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", tuple(value))
#             conn.commit()
#             cur.close()
#             conn.close()
#         except Exception, e:
#             print e.message
#             traceback.print_exc()
#         print response.xpath("//div[@class = 'main']//h1//text()").extract()[0]
#         for x in response.xpath("//div[@class = 'wrap-mc']/em/text()").extract():
#             print x.strip()
#         print response.xpath("//div[@class = 'address']//i[@class = 'icon_hf add']/parent::p//text()").extract()[0].replace(u"公司地址：", "").strip()
#         print "*****************************************************"
#
# if __name__ == "__main__":
#     pass
#
