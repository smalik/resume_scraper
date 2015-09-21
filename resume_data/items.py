# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

from scrapy.item import Item, Field

class ItemList(object):
    def __init__(self):
        self.container = []
    
    def encode(self, obj):
        return obj.__dict__

class WorkItem(Item):
    # define an object to describe particular work items from job history:
    title           = Field()
    company         = Field()
    work_location   = Field()
    start_date      = Field()
    end_date        = Field()
    description     = Field()

    def encode(self, obj):
        return obj.__dict__

class SchoolItem(Item):
    school      = Field()
    degree      = Field()
    admit_date  = Field()
    grad_date   = Field()
    locale      = Field()

    def encode(self, obj):
        return obj.__dict__

class ResultListItem(Item):
    name  = Field()
    link  = Field()

    def encode(self, obj):
        return obj.__dict__

class ResumeDataItem(ResultListItem):
    # define the fields for your resume here:
    _id         = Field()
    name        = Field()
    url         = Field()
    email       = Field()
    phone       = Field()
    headline    = Field()
    locale      = Field()
    summary     = Field()
    experience  = Field()
    education   = Field()    
    skills      = Field()
    links       = Field()

    def getName(self):
        return self.name
    
    def encode(self, obj):
        return obj.__dict__
    