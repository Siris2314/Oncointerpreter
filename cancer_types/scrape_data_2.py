import scrapy

class CancerTypeSpider(scrapy.Spider):
    name = 'cancer_types'
    start_urls = ['https://www.cancer.org/cancer/types.html']

    def parse(self, response):
        # Extracting the main cancer types
        for cancer in response.css('ul.cmp-list li.cmp-list__item'):
            cancer_name = cancer.css('span.cmp-list__item-title::text').get()
            cancer_link = response.urljoin(cancer.css('a.cmp-list__item-link::attr(href)').get())
            
            # Following link to cancer type page
            yield response.follow(cancer_link, self.parse_cancer_page, meta={'cancer_name': cancer_name})

    def parse_cancer_page(self, response):
        cancer_name = response.meta['cancer_name']
        # Extracting the sub-sections in each cancer type page
        for card in response.css('div.card__container'):
            yield {
                'cancer_name': cancer_name,
                'section_title': card.css('span.card__title::text').get(),
                'section_link': response.urljoin(card.css('a::attr(href)').get()),
            }
