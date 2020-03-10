import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spacy
import re
import sys
import en_core_web_sm
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import PCA
from matplotlib import markers,colors
from wordcloud import WordCloud


# PS this code is kinda all over the place. Different parts were ran and commeneted out after generating plots

def add_spacy_data(dataset, feature_column):
    '''
    Grabs the verb, adverb, noun, and stop word Parts of Speech (POS) 
    tokens and pushes them into a new dataset. returns an 
    enriched dataset.
    
    Parameters:
    
    dataset (dataframe): the dataframe to parse
    feature_column (string): the column to parse in the dataset.
    
    Returns: 
    dataframe
    '''
    
    verbs = []
    nouns = []
    adverbs = []
    corpus = []
    nlp = en_core_web_sm.load()
    ##
    print("Collecting Lyric Data\n----------------------")
    for i in range (0, len(dataset)):
        sys.stdout.write("\r" + "On song " + str(i+1) + " out of " + str(len(dataset)))
        sys.stdout.flush()
        song = dataset.iloc[i][feature_column]
        doc = nlp(song)
        spacy_dataframe = pd.DataFrame()
        for token in doc:
            if token.lemma_ == "-PRON-":
                    lemma = token.text
            else:
                lemma = token.lemma_
            row = {
                "Word": token.text,
                "Lemma": lemma,
                "PoS": token.pos_,
                "Stop Word": token.is_stop
            }
            spacy_dataframe = spacy_dataframe.append(row, ignore_index = True)
        verbs.append(" ".join(spacy_dataframe["Lemma"][spacy_dataframe["PoS"] == "VERB"].values))
        nouns.append(" ".join(spacy_dataframe["Lemma"][spacy_dataframe["PoS"] == "NOUN"].values))
        adverbs.append(" ".join(spacy_dataframe["Lemma"][spacy_dataframe["PoS"] == "ADV"].values))
        corpus_clean = " ".join(spacy_dataframe["Lemma"][spacy_dataframe["Stop Word"] == False].values)
        corpus_clean = re.sub("[\(\[].*?[\)\]]", ' ', corpus_clean) # this sub removes all words between [] and ()
        corpus_clean = re.sub(r'[^A-Za-z0-9]+', ' ', corpus_clean)   
        corpus.append(corpus_clean)
    dataset['Verbs'] = verbs
    dataset['Nouns'] = nouns
    dataset['Adverbs'] = adverbs
    dataset['Corpus'] = corpus
    return dataset


def top_tfidf_feats(row, features, top_n=20):
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats, columns=['features', 'score'])
    return df
def top_feats_in_doc(X, features, row_id, top_n=25):
    row = np.squeeze(X[row_id].toarray())
    return top_tfidf_feats(row, features, top_n)
def top_mean_feats(X, features, grp_ids=None, min_tfidf=0.1, top_n=25):
    if grp_ids:
        D = X[grp_ids].toarray()
    else:
        D = X.toarray()
        D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)

#x = [word,freq] , y = [word,freq]
def plotWithWords(x,y):
    n=x.shape[0]
    markerList = list(markers.MarkerStyle.markers.keys())
    normClu = colors.Normalize(np.min(n),np.max(n))
    for i in range(n):
         
         imClu = plt.scatter(
                x[i], y[i],
                marker=markerList[i % len(markerList)],
                norm=normClu, label=x[i])
    plt.colorbar(imClu)
    plt.legend().set_draggable(True)
    plt.xlabel('Word')
    plt.ylabel('Frequency')
    
def wordcloud(text, max_words):
    '''
    Wrapper around Wordcloud that increases quality, picks a specific font,
    and puts it on a white background
    '''
    
    wordcloud = WordCloud(width = 4000,
                          height = 3000,
                          background_color="white",
                          max_words = max_words                          
                         ).generate(text)
    plt.figure(figsize=(40,25))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()
    return
#give the songs of a decade, returns avg word count per song
def avgWord(songs):
    avg=0
    for lyr in songs['Lyrics']:
        avg += len(lyr)
    avg /= len(songs['Lyrics'])
    return avg
def top_feats_per_cluster(X, y, features, min_tfidf=0.1, top_n=25):
    dfs = []
    labels = np.unique(y)
    for label in labels:
        ids = np.where(y==label) 
        feats_df = top_mean_feats(X, features, ids,    min_tfidf=min_tfidf, top_n=top_n)
        feats_df.label = label
        dfs.append(feats_df)
    return dfs

def plot_tfidf_classfeats_h(dfs):
    fig = plt.figure(figsize=(12, 9), facecolor="w")
    x = np.arange(len(dfs[0]))
    for i, df in enumerate(dfs):
        ax = fig.add_subplot(1, len(dfs), i+1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xlabel("Tf-Idf Score", labelpad=16, fontsize=14)
        ax.set_title("cluster = " + str(df.label), fontsize=16)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        ax.barh(x, df.score, align='center', color='#7530FF')
        ax.set_yticks(x)
        ax.set_ylim([-1, x[-1]+1])
        yticks = ax.set_yticklabels(df.features)
        plt.subplots_adjust(bottom=0.09, right=0.97, left=0.15, top=0.95, wspace=0.52)
        plt.show()
#%%
songs = pd.read_csv('all_songs_data.csv',index_col=0)
songs = songs.dropna(subset=['Lyrics'])
songs.reset_index(inplace=True, drop=True) 
indexNames = songs[ songs['Lyrics'].map(len) > 9000 ].index
songs.drop(indexNames , inplace=True)
songs.reset_index(inplace=True, drop=True) 
del indexNames

#   manually changed subset size across 6 different Spyder consoles to emulate parrel processesing
#   I had to do it this way because Spyder seems to not do well with pythons multiproccessing library
#   This is still way faster than doing all songs in the same console. To do this you need to create an empty folder
#   and change the path to that in the file statements

d1=songs[0:1048]
d2=songs[1048:2011]
d3=songs[2011:2995]
d4=songs[2995:3960]
d5=songs[3960:4932]
d6=songs[4932:5913]

subset= d6 #this was changed for each console

songs_plus = add_spacy_data(subset, 'Lyrics')
word_counts = []
unique_word_counts = []
for i in range (0, len(songs_plus)):
    word_counts.append(len(songs_plus.iloc[i]['Lyrics'].split()))
    unique_word_counts.append(len(set(songs_plus.iloc[i]['Lyrics'].split())))
songs_plus['Word Counts'] = word_counts
songs_plus['Unique Word Counts'] = unique_word_counts

songs_plus.to_csv(r"F:\Python File Saves\song files\d6.csv") # file name was also changed for each console

del d1,d2,d3,d4,d5,d6,word_counts,unique_word_counts,subset,songs_plus,i

# At the end your new empty folder should contain 6 files and then the next part of the code joins that into 1 dataset

#%% join files
import glob
path = r'F:\Python File Saves\song files' # use your path
all_files = glob.glob(path + "/*.csv")
li = []
for file in all_files:
    df = pd.read_csv(file, index_col=None, header=0)
    li.append(df)
frame = pd.concat(li, axis=0, ignore_index=True)
frame = frame.drop(columns = ['Unnamed: 0'])
frame.reset_index(inplace=True, drop=True) 
frame.to_csv(r"F:\Python File Saves\all_songs_processed.csv")
del li,df,frame,all_files,file

songs = pd.read_csv('all_songs_processed.csv', index_col=None, header=0)
songs = songs.drop(columns = ['Unnamed: 0'])

# removing instrumental only songs
temp1 = songs.query('Corpus ==  " "')
temp2 = songs.query('Corpus ==  "Instrumental"')
temp3 = songs.query('Corpus ==  "INSTRUMENTAL"')
temp = [temp1,temp2,temp3]
for ind in temp:
    songs=songs.drop(ind.index)
instrumental_only = pd.concat(temp)
songs = songs.drop(songs[songs['Lyrics'].duplicated()==True].index) # drop duplicates, keeping the first

# manually going through data (sorted by unique words) to easily find weird, extreme mistakes caused by genius lyrics API
api_mistakes = [385,59,597,1065,3531,233,760,2155,5300,130,789,5353,5645,5816,2978,5440,5646,5630,3179,
                394,86,5544]
more_instrumentals = [208,174,1504,328]
songs = songs.drop(api_mistakes)
instrumental_only= instrumental_only.append(songs.loc[more_instrumentals])
songs = songs.drop(more_instrumentals)
songs.reset_index(inplace=True, drop=True)
del temp, temp1,temp2,temp3,ind,more_instrumentals,api_mistakes

songs.to_csv(r"F:\Python File Saves\all_songs_more_processed.csv")
instrumental_only.to_csv(r"F:\Python File Saves\some_instrumental_songs.csv")
#%% Now that most extreme cases are dealt with we can do analysis
songs = pd.read_csv('all_songs_more_processed.csv', index_col=None, header=0)
songs = songs.drop(columns = ['Unnamed: 0'])
instrumental = pd.read_csv('some_instrumental_songs.csv', index_col=None, header=0)
instrumental = instrumental.drop(columns = ['Unnamed: 0'])

#%% Analysis now
summary_dataset = pd.DataFrame()
years = songs['Year'].unique().tolist()
for i in range(0, len(years)):
    row = {
        "Year": years[i],
        "Average Words": songs['Word Counts'][songs['Year'] == years[i]].mean(),
        "Unique Words": songs['Unique Word Counts'][songs['Year'] == years[i]].mean()
    }
    summary_dataset = summary_dataset.append(row, ignore_index=True)
summary_dataset["Year"] = summary_dataset['Year'].astype(int)
char = songs.groupby('Year').count()

plt.figure(figsize=(20,10), dpi=200)
plt.plot(summary_dataset['Year'], summary_dataset['Average Words'].values, color="red", label="Average Words")
plt.plot(summary_dataset['Year'], summary_dataset['Unique Words'].values, color="green", label = "Unique Words")
plt.plot(char['Rank'], color="blue", label = "Number of Songs Per Year")
plt.xticks(summary_dataset['Year'], rotation=90)
plt.grid()
plt.legend()
plt.show()

# =============================================================================
# had to add corpus_clean = re.sub("[\(\[].*?[\)\]]", ' ', corpus_clean) to add_spacy_data
# before this, it looked like more recent songs had a lot more words and unique words, however it was disproportional to
# the real amount of change because more recent songs have better documentation on the structure of the song
# Meaning more recent songs had more things like [Break], [Chorus], (DJ), ect.
# So removing data about the structure of the song increases the accuracy of just lyrical analysis
# =============================================================================

#%% This code generates the frequencies of Nouns, Adverbs, and Verbs for every year
# Can easily make a word cloud of any years PoS with this

from collections import Counter

# frequencies is a dictionary of frequencies, max_words is the max words to be generated in the word cloud (integer)
def betterWordCloud(frequencies, max_words):   
    wc = WordCloud(width = 4000,height = 3000,max_words = max_words).generate_from_frequencies(frequencies)
    plt.figure(figsize=(40,25))
    plt.imshow(wc)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

# counts all parts of speech in the pandas dataframe under 'name' (i.e. Verbs, Nouns, Adverbs), returns Counter
def count_all_pos(songs,name):
    temp= songs[name]
    freq=[Counter()] 
    for pos in temp:
        if not isinstance(pos,str):
            continue
        counts = Counter(list(pos.split()))
        freq.append(counts)
    a= freq[0]
    for i in range(1,len(freq)):
        a+=freq[i]
    return a
allNounsCount= count_all_pos(songs,'Nouns')
allVerbsCount= count_all_pos(songs,'Verbs')
allAdverbsCount= count_all_pos(songs,'Adverbs')
    
#%% This part generates the count for all PoS by decade, resulting in a list of counters for each decade

# function that counts a years worth of pos inside songs, name = noun,verb, or adverb and songs = songs list
# input: string 'name', dataframe songs, int year
# output: a counter of the frequencies for all parts of speech for a certain year
def count_year_pos(songs,name,year):
    temp= songs[name][songs['Year'] == year]
    freq=[Counter({f'{str(year)}':1})] # adds year to the front to keep track of the year
    for pos in temp:
        if not isinstance(pos,str):
            continue
        counts = Counter(list(pos.split()))
        freq.append(counts)
    a= freq[0]
    for i in range(1,len(freq)):
        a+=freq[i]
    return a

allNounsByDecade= []
allVerbsByDecade= []
allAdverbsByDecade= []
for i in range(1959,2020):
    allNounsByDecade.append(count_year_pos(songs,'Nouns',i))
    allVerbsByDecade.append(count_year_pos(songs,'Verbs',i))
    allAdverbsByDecade.append(count_year_pos(songs,'Adverbs',i))

#betterWordCloud(dict(allVerbs[-1].most_common(200)),200)
#betterWordCloud(dict(allNouns[-1].most_common(200)),200)
#betterWordCloud(dict(allAdverbs[-1].most_common(200)),200)
del i
#%% still more processing...

# Turns a list of counters (aka allNouns,..) to a numpy array which can be used for analysis. Useful for PoSbyDecade list
# input: list of counters
# output: numpy array
def listofCounters_toNP(listofCounters):
    numpyArray= np.array(list(listofCounters[0].items()))
    for i in range(1,len(listofCounters)):
        numpyArray=np.concatenate((numpyArray,list(listofCounters[i].items())))
    return numpyArray

# this is a reversed sort for a numpy string array stored as ['word',number], but numpy only supports 1 data type for entire array
# Input: npStrList: list to be sorted, floatOrInt: True = sorted column is float, False = integer. Default is float
# Returns the sorted list in descending order
def reversedSort_npStr(npStrList,floatOrInt=True):
    if floatOrInt:
        return npStrList[np.argsort(npStrList[:,1].astype(np.float),axis=0)[:][::-1]]
    return npStrList[np.argsort(npStrList[:,1].astype(int),axis=0)[:][::-1]]

nounsNP = np.array(list(allNounsCount.items()))
verbsNP = np.array(list(allVerbsCount.items()))
adverbsNP = np.array(list(allAdverbsCount.items()))

sortedNouns = reversedSort_npStr(nounsNP)
sortedVerbs = reversedSort_npStr(verbsNP)
sortedAdverbs = reversedSort_npStr(adverbsNP)

np.savetxt('all_Nouns_Sorted.csv',sortedNouns,fmt='%s',encoding='utf-8',delimiter=',')
np.savetxt('all_Verbs_Sorted.csv',sortedVerbs,fmt='%s',encoding='utf-8',delimiter=',')
np.savetxt('all_Adverbs_Sorted.csv',sortedAdverbs,fmt='%s',encoding='utf-8',delimiter=',')
#betterWordCloud(dict(zip(sortedNouns[:,0],sortedNouns[:,1].astype(np.float))),250) for fun

#%% This part is still under construction
import mapper
from kmapper import KeplerMapper as kp
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import Isomap
from sklearn.preprocessing import MinMaxScaler

#from sklearn.feature_extraction.text import TfidfTransformer
#tf_transformer = TfidfTransformer(use_idf=False).fit(nounsNP)


maper = kp(verbose=2)
#temp = maper.fit_transform(nounsNP)
projected_X = maper.fit_transform(nounsNP[:,1],
    projection=[TfidfVectorizer(analyzer="char",
                                ngram_range=(1,6),
                                max_df=0.83,
                                min_df=0.05),
                TruncatedSVD(n_components=100,
                             random_state=1729),
                Isomap(n_components=2,
                       n_jobs=-1)],
    scaler=[None, None, MinMaxScaler()])
graph = maper.map(projected_X,
                  clusterer=AgglomerativeClustering(n_clusters=3,
                                                             linkage="complete",
                                                             affinity="cosine"),
                  overlap_perc=0.33)
vec = TfidfVectorizer(analyzer="word",
                      strip_accents="unicode",
                      stop_words="english",
                      ngram_range=(1,3),
                      max_df=0.97,
                      min_df=0.02)

interpretable_inverse_X = vec.fit_transform(nounsNP[:,0]).toarray()
interpretable_inverse_X_names = vec.get_feature_names()
html = maper.visualize(graph,
                       # inverse_X=interpretable_inverse_X,
                        #inverse_X_names=interpretable_inverse_X_names,
                        path_html="newsgroups20.html",
                        X=projected_X,
                        X_names=["ISOMAP1", "ISOMAP2"],
                        title="Newsgroups20: Latent Semantic Char-gram Analysis with Isometric Embedding")
