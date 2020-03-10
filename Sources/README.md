**Topological Data Analysis References**

I mainly used two papers as my resources for implementing TDA. These papers go through implementations of TDA on natural language as well as preparing the data using natural language processors. Most of what the TDA done in this project will be from here or a deviation from a method in here, as I'm not strong enough in group theory and topology to create my own methodologies. 

**Web Scrapper** 

I modified a web scrapper used [here ](https://github.com/sharpie-007/dataAndMusic/blob/master/49%20Years%20of%20Music%20-%20Collection%20and%20Analysis.ipynb). I also used his function to take the generated data and put it into a Pandas Dataframe, but I made my own modifications to improve it. Some of those changes included; changing progress details so it doesn't flood the output, improve regular expression removal to get rid of song structure data, and enabled it to be used in parallel processing. 
