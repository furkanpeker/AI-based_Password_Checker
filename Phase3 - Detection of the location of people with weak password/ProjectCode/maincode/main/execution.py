import os

class script:
    def execute():
        #os.system("C:/Users/fufu_/Desktop/Web_Scraping_with_Python-Scrapy/virtual_environment_library_root/Scripts/activate")
        os.chdir("C:/Users/fufu_/Desktop/ProjectCode/maincode/")
        #print(os.getcwd())
        os.system("scrapy crawl quotes -o items.csv")

