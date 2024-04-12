import scrapy

class MdandersonCancerTypesSpider(scrapy.Spider):
    name = "mdanderson_cancer_types"
    allowed_domains = ["www.mdanderson.org"]
    start_urls = ["https://www.mdanderson.org/patients-family/diagnosis-treatment/cancer-types.html"]

    def parse(self, response):
        # Loop through each letter section
        for letter_section in response.css('div.letter-section.link-list'):
            # Extract from the left side
            for li in letter_section.css('.item-container-first > li.glossary-search-result'):
                link = li.css('a::attr(href)').get()
                name = li.css('.glossary-search-title p::text').get().strip()
                if link and name:
                    yield {
                        'name': name,
                        'link': response.urljoin(link),
                    }
            # Extract from the right side
            for li in letter_section.css('.item-container-last > li.glossary-search-result'):
                link = li.css('a::attr(href)').get()
                name = li.css('.glossary-search-title p::text').get().strip()
                if link and name:
                    yield {
                        'name': name,
                        'link': response.urljoin(link),
                    }
