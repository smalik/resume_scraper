import csv
import requests
import json
import simplejson as sj
import hashlib
import msgpack

import redis
import redis_collections as rc
import redis_wrap as rw
import hiredis

from bs4 import BeautifulSoup, Tag, NavigableString, Comment
from bs4.element import NavigableString

from resume_data.items import ItemList, ResumeDataItem, ResultListItem, WorkItem, SchoolItem
from scrapy.selector import HtmlXPathSelector, Selector
from scrapy.contrib.exporter import XmlItemExporter

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html

class IndeedResumesPipeline(object):

    def __init__(self):
        #   Define redis conection variables
        self.POOL            = redis.ConnectionPool(host='plytos.com', port=6379, db=1)
        self.BUCKET          = 1000
        self.r = redis.Redis(connection_pool=self.POOL)
        #self.r = redis.StrictRedis(host='plytos.com', port=6379, db=0)
        self.pipe = self.r.pipeline(transaction=True)
        
    def __index__(self):
        self.digest = {}
    
    ''' Generates a hash string using a supplied URL.  Intended to serve as primary key in storing job data in Redis    '''
    def getUniqueKey(self, url):
        hash = hashlib.md5()
        #hash = hashlib.sha1()
        hash.update(url)
        return hash.hexdigest()

    def process_item(self, item, spider):
        url     =   item['url']
        _id     =   self.getUniqueKey(url)
        self.r.hset(_id, 'url', url)

        name            =   item['name']
        self.r.hset(_id, 'name', name)

        headline        =   item['headline']
        self.r.hset(_id, 'headline', headline)

        locale          =   item['locale']
        self.r.hset(_id, 'locale', locale)

        summary         =   item['summary']
        self.r.hset(_id, 'summary', summary)

        education       =   item['education']
        self.r.hset(_id, 'education', education)

        skills          =   item['skills']
        self.r.hset(_id, 'skills', skills)

        links           =   item['links']
        self.r.hset(_id, 'links', links)

        experience      =   item['experience']
        self.r.hset(_id, 'experience', experience)
        
        return None


class ResumesNgramPipeline(object):

    def __init__(self):
        pass
    
    def find_ngrams(input_list, n):
        return zip(*[input_list[i:] for i in range(n)])
    
    def process_item(self):
        
        keys    =   self.r
        pass