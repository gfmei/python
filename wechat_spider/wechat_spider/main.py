#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: gfmei
# finish date: 20160302
from twisted.internet import reactor
from scrapy.crawler import CrawlerRunner
from scrapy.utils.project import get_project_settings
runner = CrawlerRunner(get_project_settings())

# 'wechat_spider' is the name of one of the spiders of the project.
d = runner.crawl('anyv')
d.addBoth(lambda _: reactor.stop())
# the script will block here until the crawling is finished
reactor.run()