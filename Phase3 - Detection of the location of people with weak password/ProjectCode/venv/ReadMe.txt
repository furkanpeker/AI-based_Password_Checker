--> What is a virtual enviroment (venv) and why do you need to install one?

A venv basically isolates your scraping project from the rest of the computer, 
meaning it provides the development enviroment that normally the OS should provide.
For example, if you install any kind of package inside your main scrapy folder, 
it's not going to be installed to rest of the computer. 
And on the contrary if you install some kind of package in the rest of the computer, 
there isn't going to installed inside this scrapy main folder. From this aspect it enable the isolation! 

Summary, if we install a scrapy package inside the scrapy folder with the virtual enviroment activated, 
then scrapy is just going to be installed inside this scrappy folder and not in the rest of our computer.