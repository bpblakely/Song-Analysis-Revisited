**Song Scraper.py**

  This is how the data set is generated. I used the web scraper built [here](https://github.com/sharpie-007/dataAndMusic/blob/master/49%20Years%20of%20Music%20-%20Collection%20and%20Analysis.ipynb)
 which was slightly broken, so it needed minor fixes to work. This outputs a csv and json file containing the song data of the top 100 songs from the past 60 years. It takes about 6 hours to run, but could be done way quicker with parallel processing.
 
 ![](https://i.imgur.com/mDZkDUI.png)

**SongAnalysis.py**


This is our main program. Note that this was named after we finally got some sort of tangable results. Still some works needs to be done. Some of the commented out pieces of code were used to generate graphs, but don't need to be ran all the time.
