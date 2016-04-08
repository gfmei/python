#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: gfmei
# finish date: 20160302
from twisted.internet import reactor
from scrapy.crawler import CrawlerRunner
from scrapy.utils.project import get_project_settings
runner = CrawlerRunner(get_project_settings())
#scrapy crawl baidu
# 'followall' is the name of one of the spiders of the project.
d = runner.crawl('cn360cn')
d.addBoth(lambda _: reactor.stop())
reactor.run() # the script will block here until the crawling is finished