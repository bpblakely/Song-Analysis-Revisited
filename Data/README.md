All_songs is the data generated after ~6 hours of scraping. This data set is the top 100 songs of the year for the last 60 years. 
The data set is available as either a json or csv file. Lyrics were generated with Genius API.

**Data Descriptions**
--------------------------------------------------------------------------------------------------------------------------------------
**all_songs_data**

Data set size: ( 6100 x 12 )

Each row describes a song with 12 variables.

The variables in order:

1. Album
2. Album URL
3. Artist
4. Featured Artists 
5. Lyrics
6. Media Link (i.e. youtube link)
7. Song Ranking at end of year
8. Release Date (year-month-day)
9. Song Title
10. Lyrics URL (source for lyrics)
11. Writers (Genius API related)
12. Year Released

Note: Genius API isn't perfect, so there's some bad data in the mix.

**all_songs_processed**

This file contains the result of Spacy NLP on the data to extract Parts of Speech from the lyrics. This file is all_songs_data data frame with 6 extra columns appended on to it which are described as follows:
1. Verbs
     * This column contains the verbs extracted from a songs lyrics by Spacy
2. Nouns
     * This column contains the nouns extracted from a songs lyrics by Spacy
3. Adverbs
     * This column contains the adverbs extracted from a songs lyrics by Spacy
4. Corpus
     * The raw text after processing. This text is what Verbs, Nouns, and Adverbs are generated from
5. Word Counts
      * An integer representing the number of words in a song
6. Unique Words
     * An integer representing the number of unique words in a song 

Note: The preprocessing for this step to generate Corpus removed any words between parethensis ( ) and brackets [ ]. This was done to ensure that things like [Verse] and such were removed as it skewed the word count significantly.


 **all_songs_more_processed**  	(in all_songs_processed.zip)
 
This file is just an extension of all_songs_processed (see above). The difference is that this file attempts to scrub some of the noticeably bad data points, removes instrumental music (seperated into different file), and drops songs with no corpus.
    
**some_instrumental_songs** 	(in all_songs_processed.zip)

This file is the file generated from all_songs_more_processed. It is just a filtering of instrumental songs into a different file. The goal was to preserve instrumental only music if the data was relevant for different analysis. 
	
Instrumental music was filted by the following;
1. Corpus ==  " " 
      * If corpus is empty
2. Corpus ==  "Instrumental"
3. Corpus ==  "INSTRUMENTAL"
4. Manually afterwards
     * Some manual cases included weird ascii characters like a music note. Only a couple were manually filtered
  
**all_(PoS)_Sorted** 
 
These template relates to; all_Verbs_Sorted, all_Nouns_Sorted, and all_Adverbs_Sorted . These files are just a simple list of their relative PoS and a count of how many times they occured in all lyrics for every song in all_songs_more_processed. The list is sorted by value and is in descending order.

These files are designed to be imported by

	numpy.loadtxt('filename',delimeter=',', dtype='str', encoding='utf-8')
