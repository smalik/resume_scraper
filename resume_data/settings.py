# Scrapy settings for craigslist_jobs project
#
# For simplicity, this file contains only the most important settings by
# default. All the other settings are documented here:
#
#     http://doc.scrapy.org/en/latest/topics/settings.html
#

BOT_NAME = 'resume_data'

#DOWNLOADER_MIDDLEWARES = {
#        'scrapy.contrib.downloadermiddleware.useragent.UserAgentMiddleware' : None,
#        'resume_data.rotate_useragent.RotateUserAgentMiddleware' :400
#    }
SPIDER_MODULES = ['resume_data.spiders']
NEWSPIDER_MODULE = 'resume_data.spiders'

#ITEM_PIPELINES = {
#    'resume_data.pipelines.IndeedResumesPipeline': 100
#}

# Crawl responsibly by identifying yourself (and your website) on the user-agent
#USER_AGENT = 'reume_data (+http://www.yourdomain.com)'

DEPTH_LIMIT = 50

MEMUSAGE_REPORT = True

AJAXCRAWL_ENABLED = True
COOKIES_ENABLED = False
DOWNLOAD_DELAY = 2

LOG_ENABLED = True
