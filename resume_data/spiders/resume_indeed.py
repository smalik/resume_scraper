#!/usr/bin/env python
import types
import time
from datetime import date, datetime, timedelta

import requests
import msgpack

from scrapy.http import Request
from scrapy.contrib.spiders import CrawlSpider, Rule
from scrapy.contrib.linkextractors.sgml import SgmlLinkExtractor
from scrapy.selector import HtmlXPathSelector, Selector
from resume_data.items import ResumeDataItem, ResultListItem, WorkItem, SchoolItem, ItemList

from bs4 import BeautifulSoup, Tag, NavigableString, Comment
from bs4.element import NavigableString

class ResumeIndeedSpider(CrawlSpider):
    name = "indeed_resume"
    allowed_domains = ["indeed.com"]
    start_urls = ['http://www.indeed.com/resumes/statistics-python',
                  'http://www.indeed.com/resumes/data-scientist',
                  'http://www.indeed.com/resumes/python',
                  'http://www.indeed.com/resumes/java',
                  'http://www.indeed.com/resumes/SAS',
                  'http://www.indeed.com/resumes/javascript',
                  'http://www.indeed.com/resumes/marketing-analyst']

    #def __init__(self, filename=None):
            #self.unis    =   list()

    
    rules = (Rule (SgmlLinkExtractor(restrict_xpaths = ('//a[@class="instl confirm-nav next"]')), callback = "parse_item", follow = True),)

    
    def parse_item(self, response):
        hxs     =   Selector(response)
        digest  =   hxs.xpath('//ol[@class="resultsList"]')
        records =   ResumeDataItem()
        
        url_prefix = 'http://www.indeed.com'

        resume_links   =   digest.xpath('//li[@class="sre"]//div[@class="sre-entry"]')
        names   =   digest.xpath('//a[@target="_blank"]/text()').extract()
        links   =   digest.xpath('//a[@target="_blank"]/@href').extract()
                
        for name, link in zip(names,links):
            if name not in 'Feedback':
                records['name'] =   name
                records['link'] =   url_prefix+link
                yield Request(records['link'], meta={'item': records}, callback= self.parse_node)


    def parse_node(self, response):
        hxs     =   Selector(response)
        records =   ResumeDataItem()
        
#        name    =   hxs.xpath('/text()').extract()
        name        =   hxs.xpath('//h1[@id="resume-contact"]/text()').extract()
        headline    =   hxs.xpath('//h2[@id="headline"]/text()').extract()
#        locale      =   hxs.xpath('//div[@class="addr" and @itemprop="address"]//p//text()').extract()
        rlocale      =   hxs.xpath('//p[@id="headline_location" and @class="locality"]//text()').extract()
        summary     =   hxs.xpath('//p[@id="res_summary" and @class="summary"]/text()').extract()
        skills      =   list()
        skill       =   hxs.xpath('//div[@id="skills-items" and @class="items-container"]//p//text()').extract()
        if len(skill) != 0:
            skills.append(''.join(skill).encode('utf-8'))        
        skill       =   hxs.xpath('//div[@id="additionalinfo-section" and @class="last"]//div[@class="data_display"]//p//text()').extract()
        if len(skill) != 0:
            skills.append(''.join(skill).encode('utf-8'))        
        
        resume_links    =   list()
        links       =   hxs.xpath('//div[@id="link-items" and @class="items-container"]//p//text()').extract()
        for link in links:
            resume_links.append(''.join(link).encode('utf-8'))

        workHistory =   ItemList()
        experience  =   hxs.xpath('//div[@id="work-experience-items"]/div')
        for elem in experience:
            item = elem.xpath('div')
            for entry in item:
                workEntry   =   WorkItem()

                title       =   entry.xpath('p[@class="work_title title"]//text()').extract()
                workEntry['title']  =   ''.join(title).encode('utf-8')

                company     =   entry.xpath('div[@class="work_company"]/span/text()').extract()
                workEntry['company']=   ''.join(company).encode('utf-8')

                location    =   entry.xpath('div[@class="work_company"]/div[@class="inline-block"]/span/text()').extract()
                workEntry['work_location']  =   ''.join(company).encode('utf-8')

                dates       =   entry.xpath('p[@class="work_dates"]//text()').extract()
                dates_str   =   ''.join(dates).encode('utf-8').split(' to ')
                if len(dates) > 0:
                    if dates_str[0]:
                        workEntry['start_date'] =   dates_str[0]
                    if dates_str[1]:
                        workEntry['end_date']   =   dates_str[1]
                else:
                    workEntry['start_date'] =   'NULL'
                    workEntry['end_date']   =   'NULL'
                    

                description =   entry.xpath('p[@class="work_description"]//text()').extract()
                workEntry['description']    =   ''.join(description).encode('utf-8')

                workHistory.container.append(workEntry)
                
        eduHistory =   ItemList()
        education  =   hxs.xpath('//div[@id="education-items" and @class="items-container"]/div')
        for elem in education:
            item = elem.xpath('div')
            for entry in item:
                eduEntry    =   SchoolItem()

                degree      =   entry.xpath('p[@class="edu_title"]/text()').extract()
                degree      =   ''.join(degree).encode('utf-8')
                eduEntry['degree']  =   degree

                school      =   entry.xpath('div[@class="edu_school"]/span//text()').extract()
                school      =   ''.join(school).encode('utf-8')
                eduEntry['school']  =   school

                locale      =   entry.xpath('span[@itemprop="addressLocality"]/text()').extract()
                locale      =   ''.join(locale).encode('utf-8')
                eduEntry['locale']  =   locale
                
                grad_date   =   entry.xpath('p[@class="edu_dates"]/text()').extract()
                dates_str   =   ''.join(grad_date).encode('utf-8').split(' to ')
                if len(grad_date) > 0:
                    if len(dates_str) == 2:
                        if dates_str[0]:
                            eduEntry['admit_date']  =   dates_str[0]
                        try:
                            if dates_str[1]:
                                eduEntry['grad_date']   =   dates_str[1]
                        except:
                            pass
                    elif len(dates_str) == 1:
                        if dates_str[0]:
                            eduEntry['grad_date']  =   dates_str[0]
                            eduEntry['admit_date'] =   'NULL'
                else:
                    eduEntry['admit_date']  =   'NULL'
                    eduEntry['grad_date']   =   'NULL'

                eduHistory.container.append(eduEntry)

        records['url']      =   response.url
        records['name']     =   ''.join(name).encode('utf-8')
        records['headline'] =   msgpack.packb(''.join(headline).encode('utf-8'))
        records['locale']   =   ''.join(rlocale).encode('utf-8')
        records['summary']  =   msgpack.packb(''.join(summary).encode('utf-8'))
        records['skills']   =   msgpack.packb(skills)
        records['links']    =   resume_links
        #records['experience']   =   msgpack.packb(workHistory, default=workHistory.encode)
        records['experience'] = workHistory
        records['education']    =   msgpack.packb(eduHistory, default=eduHistory.encode)
        #records['experience']   =   workHistory
        #records['education']    =   eduHistory

        return records
    
