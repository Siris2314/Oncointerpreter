import scrapy

class CancerTypeSpider(scrapy.Spider):
    name = 'cancer_types'
    start_urls = ['https://www.mskcc.org/cancer-care/types']

    def parse(self, response):
        # Extracting the content using css selectors
        for cancer in response.css('ul.msk-list.msk-list--unordered li.msk-list-item'):
            yield {
                'name': cancer.css('a::text').get(),
                'link': response.urljoin(cancer.css('a::attr(href)').get()),
            }