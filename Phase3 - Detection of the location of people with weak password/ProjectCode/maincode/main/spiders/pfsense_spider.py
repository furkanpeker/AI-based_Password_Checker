import scrapy
from ..items import MainItem
# To crawl (execute) the spider : scrapy crawl <spider name>

class PfsenseSpider(scrapy.Spider): # Here our class QuoteSpider extends / inherit from the scrapy and inside scrapy extends / inherits from Spider class
    name = 'quotes' # inherited from scrapy.Spider class
    start_urls = [ # inherited from scrapy.Spider class
        # urls of pages we want to scrap
        'http://quotes.toscrape.com/'
    ]
    

    # now let's parse the web page into fields that are going to columns of the output table
    def parse(self, response):
        # create our item instance invoking our item class in items.py
        items = MainItem()
        '''
        The reason of putting the data into containers is that it's always a good idea to move this scrap data to temporary location called containers 
        and then store them inside the database. These containers where we are storing the extracted data are called as 'items'.
        To do this, we will use class in the 'items.py' file to create our item containers, and the related item class for this job (for this example, the QuotetutorialItem) automatically created for us by scrapy when we created the scrapy project at the beginnig. 
        Go to 'items.py' and customise the class content according to you, and then import the class inside items.py into related spider python file (here the 'quotes_spider.py')).
        '''

        # extraction phase of code
        '''
        # here the response is which basically contains the source code of our web site that we want to scrap, 
        # so the url/urls on the 'start_url' list is going to send the source code within this 'response' variable
        # summary we said creating this parse function that "Hey spider, go to the source code of web page that we are looking at, and look for 'title' tag. After you've found the title tag, just extract it and after extract it, yield them / return it (for cases just have one) to show us in an output file!"

        title = response.css('title::text').extract() # it'll fetch in other word extract just the 'title' tag from the source code (html code) of web site
        yield {'titletext' : title}
        
        
        Yield statement is the iterator version of the return statement known as we all, so the yield statement suspends a function’s execution and sends a value back to the caller, 
        but retains enough state to enable the function to resume where it left off. 
        When the function resumes, it continues execution immediately after the last yield run. 
        This allows its code to produce a series of values over time, rather than computing them at once and sending them back like a list.
        Therefore we used the yield here as dictionary to be able to return the data as sets of key-values.
        As you suggest the 'titletext' key is the first and one column of our output table if we considered we take output as .csv file.

        '''
        ''' 
        # the general syntax style:
        all_div_quotes = response.css('div.quote') # we're fetching all division tags on the page's source code, and assign them into 'all_div_quotes' variable. Thus fetching only tags in the division tags of quotes on the page instead of in all source code. 

        # now let's extract what we want:
        title = all_div_quotes.css('span.text::text').extract()#[0]
        author = all_div_quotes.css('.author::text').extract()#[0]
        tag = all_div_quotes.css('.tag::text').extract()#[:4]
        
        
        yield {
            'titletext' : title,
            'authortext' : author,
            'tagstext' : tag
        }
        '''
        #if we want to extract the texts session by session:
        all_div_quotes = response.css('div.quote')
        for quotes in all_div_quotes:
            title = quotes.css('span.text::text').extract()
            author = quotes.css('.author::text').extract()
            tag = quotes.css('.tag::text').extract()
            '''
            yield {
                'titletext' : title,
                'authortext' : author,
                'tagstext' : tag
            }
            '''
        
            # to be able to use the 'items' container instead of using the dictionary mechanism of python, match our tag items with our related item of 'items' object:
            items['title'] = title
            items['author'] = author
            items['tag'] = tag

            yield items



'''
Examples of extracting data:

>>> response.xpath("//span[@class='text']/text()").extract()
['“The world as we have created it is a process of our thinking. It cannot be changed without changing our thinking.”', '“It is our choices, Harry, that show what we truly are, far more than our abilities.”', '“There are only two ways to live your life. One is as though nothing is a miracle. The other is as though everything is a miracle.”', '“The person, be it gentleman or lady, who has not pleasure in a good novel, must be intolerably stupid.”', "“Imperfection is beauty, madness is genius and it's better to be absolutely ridiculous than absolutely boring.”", '“Try not to become a man of success. Rather become a man of value.”', '“It is better to be hated for what you are than to be loved for what you are not.”', "“I have not failed. I've just found 10,000 ways that won't work.”", "“A woman is like a tea bag; you never know how strong it is until it's in hot water.”", '“A day without sunshine is like, you know, night.”']

>>> response.xpath("//title/text()").extract()
['Quotes to Scrape']

>>> response.xpath("/html/body/div/div[2]/div[1]/div[1]/span[1]/text()").extract()
['“The world as we have created it is a process of our thinking. It cannot be changed without changing our thinking.”']

>>> response.xpath("/html/body/div/div[2]/div[1]/div[1]/span[1]").extract()        
['<span class="text" itemprop="text">“The world as we have created it is a process of our thinking. It cannot be changed without changing our thinking.”</span>']

>>> response.xpath("/html/body/div/div[2]/div[1]/div[1]/span[1]/text()").extract() 
['“The world as we have created it is a process of our thinking. It cannot be changed without changing our thinking.”']

>>> response.xpath("//span[@class='text']/text()")[2].extract()  
'“There are only two ways to live your life. One is as though nothing is a miracle. The other is as though everything is a miracle.”'

>>> response.xpath("//span[@class='text']/text()")[1].extract() 
'“It is our choices, Harry, that show what we truly are, far more than our abilities.”'

>>> response.xpath("//span[@class='text']/text()")[0].extract() 
'“The world as we have created it is a process of our thinking. It cannot be changed without changing our thinking.”'

'''

'''
Examples of extracting by combining both CSS and XPATH selectors :

>>> response.css("li.next a").xpath("@href").extract()   
['/page/2/']

--> to get all 'href' tags on the source code of the page:
>>> response.css("a").xpath("@href").extract()   
['/', '/login', '/author/Albert-Einstein', '/tag/change/page/1/', '/tag/deep-thoughts/page/1/', '/tag/thinking/page/1/', '/tag/world/page/1/', '/author/J-K-Rowling', '/tag/abilities/page/1/', '/tag/choices/page/1/', '/author/Albert-Einstein', '/tag/inspirational/page/1/', '/tag/life/page/1/', '/tag/live/page/1/', '/tag/miracle/page/1/', '/tag/miracles/page/1/', '/author/Jane-Austen', '/tag/aliteracy/page/1/', '/tag/books/page/1/', '/tag/classic/page/1/', '/tag/humor/page/1/', '/author/Marilyn-Monroe', '/tag/be-yourself/page/1/', '/tag/inspirational/page/1/', '/author/Albert-Einstein', '/tag/adulthood/page/1/', '/tag/success/page/1/', '/tag/value/page/1/', '/author/Andre-Gide', '/tag/life/page/1/', '/tag/love/page/1/', '/author/Thomas-A-Edison', '/tag/edison/page/1/', '/tag/failure/page/1/', '/tag/inspirational/page/1/', '/tag/paraphrased/page/1/', '/author/Eleanor-Roosevelt', '/tag/misattributed-eleanor-roosevelt/page/1/', '/author/Steve-Martin', '/tag/humor/page/1/', '/tag/obvious/page/1/', '/tag/simile/page/1/', '/page/2/', '/tag/love/', '/tag/inspirational/', '/tag/life/', '/tag/humor/', '/tag/books/', '/tag/reading/', '/tag/friendship/', '/tag/friends/', '/tag/truth/', '/tag/simile/', 'https://www.goodreads.com/quotes', 'https://scrapinghub.com']

'''

