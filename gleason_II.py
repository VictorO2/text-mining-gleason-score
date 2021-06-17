# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 12:45:57 2017

@author: VictorO
"""

#importing the libraries
print('importing required packages...')
import time, re, json, warnings, psycopg2, pandas as pd, os, matplotlib.pyplot as plt, seaborn as sns
from sqlalchemy import create_engine
from wordcloud import WordCloud, STOPWORDS
import unicodedata
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
porter_stemmer = PorterStemmer()
from IPython.core.display import display, HTML

start = time.time()
startptd = time.strftime('%X %x %Z')
print('The program start time and Date','\n',startptd)

#seting the columns views to full
pd.set_option('display.max_columns', None)  
pd.set_option('display.max_rows', None)
sns.set(font_scale = 2)
sns.set_style("whitegrid", {'axes.grid' : False})
warnings.filterwarnings('ignore')
#sns.set(font_scale = 2)
#sns.set_style("white")

print('setting up working directory...')
#setting the working directory
os.chdir("M:\\gleason score\\Gleasen_scores\\")

print('reading the dataset...')
#reading the datasets
#loading the file to memory
conn = psycopg2.connect(user="postgres", password="XXXXX",
                                  host="000.0.0.0", port="0000", database="XXXX")
cursor = conn.cursor()
sql_cmd1 = "SELECT dummy_id, result_text FROM sampled_data_1000"
cursor.execute(sql_cmd1)
dfA = pd.read_sql(sql_cmd1, conn)
#closing the connection
cursor.close()
conn.close()
print('data read successfully and connection closed...')
print(len(dfA))
print(dfA.head())

print('printing wordcloud before cleaning')
wordcloud = WordCloud(max_words=400, width = 3000, height = 2000, random_state=42, 
                      background_color='black', colormap='tab20c', collocations=True, 
                      stopwords = STOPWORDS).generate(str(dfA['result_text']))
print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad = 0)
fig.savefig("cloud_before_cleaning.png", dpi=900)
fig.savefig("cloud_before_cleaning.pdf", dpi=900)
plt.show()

#word counts before cleaning
def uniqueWords(X):
    X = X.split(' ')
    X = set(X)
    X = len(X)
    return X
dfA['charCount_before']   = dfA['result_text'].str.len()
dfA['wordCount_before']   = dfA['result_text'].str.split(' ').str.len()
dfA['uniqueWords_before'] = dfA['result_text'].apply(uniqueWords)

#removing extra lines and tabs in the notes
dfA['result_text'] = dfA['result_text'].replace('\n',' ', regex=True)
dfA['result_text'] = dfA['result_text'].replace('\t',' ', regex=True)
dfA['result_text'] = dfA['result_text'].replace('\:',' ', regex=True)
dfA['result_text'] = dfA['result_text'].replace('\(',' ', regex=True)
dfA['result_text'] = dfA['result_text'].replace('\)',' ', regex=True)
dfA['result_text'] = dfA['result_text'].replace('\=',' ', regex=True)
dfA['result_text'] = dfA['result_text'].replace('\¿',' ', regex=True)
dfA['result_text'] = dfA['result_text'].replace('\-',' ', regex=True)
dfA['result_text'] = dfA['result_text'].replace('\s{2,}',' ', regex=True)

print('performing stemming and tokenization')
def stem_sentences(sentence):
    tokens = sentence.split()
    stemmed_tokens = [porter_stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

dfA['result_text'] = dfA['result_text'].apply(stem_sentences)
dfA['result_text'] = dfA['result_text'].apply(lambda x : filter(None,x.split(" ")))
dfA['result_text'] = dfA['result_text'].apply(lambda x : [porter_stemmer.stem(y) for y in x])
dfA['result_text'] = dfA['result_text'].apply(lambda x : " ".join(x))

print('calling the stopwords list and adding a few words to the list')     
eng_stopwords = set(stopwords.words("english"));
#add words that aren't in the NLTK stopwords list
#more_stop_words = ['tel','telephone','fax']
#eng_stopwords = eng_stopwords.union(more_stop_words)    

print('removing the stopwords from the corpus') 
for row in dfA.index:
    text_revi = dfA.loc[row, 'result_text'].split()
    dfA.loc[row, 'result_text']  = ' '.join([word for word in text_revi if word not in eng_stopwords]) 
    
print('picking up only words that are a to z, A to Z, numbers and a few signs then converting the words to lowercase...')
pattern_to_find = "[^a-zA-Z0-9\-\+\,\;\.\/' ]";
pattern_to_repl = "";
for row in dfA.index:
    dfA.loc[row, 'result_text'] = re.sub(pattern_to_find, pattern_to_repl, dfA.loc[row, 'result_text']).lower()    

#correcting gleason
dfA['result_text'] = dfA['result_text'].replace('gleeson','gleason', regex=True)
dfA['result_text'] = dfA['result_text'].replace('glison','gleason', regex=True)

print('printing wordcloud after cleaning')
wordcloud = WordCloud(max_words=400, width = 3000, height = 2000, random_state=42, 
                      background_color='black', colormap='tab20c', collocations=True, 
                      stopwords = STOPWORDS).generate(str(dfA['result_text']))
print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad = 0)
fig.savefig("cloud_after_cleaning.png", dpi=900)
fig.savefig("cloud_after_cleaning.pdf", dpi=900)
plt.show()

#word counts after cleaning
def uniqueWords(X):
    X = X.split(' ')
    X = set(X)
    X = len(X)
    return X
dfA['charCount_after']   = dfA['result_text'].str.len()
dfA['wordCount_after']   = dfA['result_text'].str.split(' ').str.len()
dfA['uniqueWords_after'] = dfA['result_text'].apply(uniqueWords)

#creating a dataframe for the stats
dfA_subA = pd.concat([dfA['charCount_before'], dfA['charCount_after'], dfA['wordCount_before'], 
                      dfA['wordCount_after'], dfA['uniqueWords_before'], dfA['uniqueWords_after']], axis = 1)

#percentage change in the cleaning
dfA_subA['percentage_change'] = ((dfA_subA['wordCount_before']-dfA_subA['wordCount_after'])/dfA_subA['wordCount_before'])*100

sns.distplot(dfA_subA.percentage_change.astype(float).dropna())
fig = plt.figure(1)
#plt.title('Percentage change after Pre-processing')
plt.xlabel('Percentage Change')
plt.ylabel('Densities weights')
fig.savefig("preprocessing_percentage_change.png", dpi=900)
fig.savefig("preprocessing_percentage_change.pdf", dpi=900)
plt.show()

sub_table = round(dfA_subA.describe().T, 2)
print(sub_table)

#save the sub table
sub_table.to_excel('cleaning_description.xlsx')

# KDE plots for each species
fig = plt.figure(figsize=(18,12))
sns.set(font_scale = 2)
sns.set_style("white")
sns.kdeplot(data=dfA_subA['charCount_before'], label="Character count before pre-processing", shade=True)
sns.kdeplot(data=dfA_subA['wordCount_before'], label="Word count before pre-processing", shade=True)
sns.kdeplot(data=dfA_subA['uniqueWords_before'], label="Unique word count before pre-processing", shade=True)
plt.legend(fontsize= 25)
plt.xlabel('Character, word and unique word count before pre-processing', fontsize = 25)
plt.ylabel('Densities weights', fontsize = 25)
fig.tight_layout()
fig.savefig("char_word_unique_counts_before.png", dpi=900)
fig.savefig("char_word_unique_counts_before.pdf", dpi=900)
plt.show()

# KDE plots for each species
fig = plt.figure(figsize=(18,12))
sns.kdeplot(data=dfA_subA['charCount_after'], label="Character count after pre-processing", shade=True)
sns.kdeplot(data=dfA_subA['wordCount_after'], label="Word count after pre-processing", shade=True)
sns.kdeplot(data=dfA_subA['uniqueWords_after'], label="Unique word count after pre-processing", shade=True)
plt.legend(fontsize= 25)
plt.xlabel('Character, word and unique word count after pre-processing', fontsize = 25)
plt.ylabel('Densities weights', fontsize = 25)
fig.tight_layout()
fig.savefig("char_word_unique_counts_after.png", dpi=900)
fig.savefig("char_word_unique_counts_after.pdf", dpi=900)
plt.show()

# Histograms for each species
fig = plt.figure(figsize=(18,6))
sns.distplot(a=dfA_subA['charCount_before'], label="Character count before", kde=False)
sns.distplot(a=dfA_subA['charCount_after'], label="Character count after", kde=False)
sns.distplot(a=dfA_subA['wordCount_before'], label="Word count before", kde=False)
sns.distplot(a=dfA_subA['wordCount_after'], label="Word count after", kde=False)
sns.distplot(a=dfA_subA['uniqueWords_before'], label="Unique word count before", kde=False)
sns.distplot(a=dfA_subA['uniqueWords_after'], label="Unique word count after", kde=False)
plt.title("Distribution of Characters, Words and Unique words")
plt.legend()
plt.xlabel('Character, word and unique word count before and after pre-processing', fontsize = 25)
plt.ylabel('Densities counts', fontsize = 25)
fig.tight_layout()
fig.savefig("char_word_unique_counts_bef_after.png", dpi=900)
fig.savefig("char_word_unique_counts_bef_after.pdf", dpi=900)
plt.show()

fig = plt.figure(figsize=(32,12))
ax1 = plt.subplot(3, 2, 1)
sns.kdeplot(data=dfA_subA['charCount_before'], label="Character count before pre-processing")
plt.xlabel('Character count before pre-processing')
ax2 = plt.subplot(3, 2, 2)
sns.kdeplot(data=dfA_subA['charCount_after'], label="Character count after pre-processing")
plt.xlabel('Character count after pre-processing')
ax2 = plt.subplot(3, 2, 3)
sns.kdeplot(data=dfA_subA['wordCount_before'], label="Word count before pre-processing")
plt.xlabel('Word count before pre-processing')
ax2 = plt.subplot(3, 2, 4)
sns.kdeplot(data=dfA_subA['wordCount_after'], label="Word count after pre-processing")
plt.xlabel('Word count after pre-processing')
ax2 = plt.subplot(3, 2, 5)
sns.kdeplot(data=dfA_subA['uniqueWords_before'], label="Unique Word count before pre-processing")
plt.xlabel('Unique word count before pre-processing')
ax2 = plt.subplot(3, 2, 6)
sns.kdeplot(data=dfA_subA['uniqueWords_after'], label="Unique Word count after pre-processing")
plt.xlabel('Unique word count after pre-processing')
fig.tight_layout()
fig.savefig("kdeplot_cleaning_characteristics1.png", dpi=900)
fig.savefig("kdeplot_cleaning_characteristics1.pdf", dpi=900)
plt.show()

fig = plt.figure(figsize=(32,12))
ax1 = plt.subplot(3, 2, 1)
sns.distplot(a=dfA_subA['charCount_before'], label="Character count before pre-processing")
plt.xlabel('Character count before pre-processing')
ax2 = plt.subplot(3, 2, 2)
sns.distplot(a=dfA_subA['charCount_after'], label="Character count after pre-processing")
plt.xlabel('Character count after pre-processing')
ax2 = plt.subplot(3, 2, 3)
sns.distplot(a=dfA_subA['wordCount_before'], label="Word count before pre-processing")
plt.xlabel('Word count before pre-processing')
ax2 = plt.subplot(3, 2, 4)
sns.distplot(a=dfA_subA['wordCount_after'], label="Word count after pre-processing")
plt.xlabel('Word count after pre-processing')
ax2 = plt.subplot(3, 2, 5)
sns.distplot(a=dfA_subA['uniqueWords_before'], label="Unique Word count before pre-processing")
plt.xlabel('Unique word count before pre-processing')
ax2 = plt.subplot(3, 2, 6)
sns.distplot(a=dfA_subA['uniqueWords_after'], label="Unique Word count after pre-processing")
plt.xlabel('Unique word count after pre-processing')
fig.tight_layout()
fig.savefig("distplot_cleaning_characteristics1.png", dpi=900)
fig.savefig("distplot_cleaning_characteristics1.pdf", dpi=900)
plt.show()

#creating a pairplot
sns.pairplot(dfA_subA, vars=["charCount_before", "charCount_after", "wordCount_before", "wordCount_after",
                            "uniqueWords_before", "uniqueWords_after"],
             diag_kind = 'kde',
             plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'},
             size = 4)
fig = plt.figure(1)
#fig.tight_layout()
fig.savefig("pair_plot_cleaning_characteristics1.png", dpi=900)
fig.savefig("pair_plot_cleaning_characteristics1.pdf", dpi=900)
plt.show()


print('getting word counts for unigrams and bi-grams')
def basic_clean(text):
  """
  A simple function to clean up the data. All the words that
  are not designated as a stop word is then lemmatized after
  encoding and basic regex parsing are performed.
  """
  wnl = nltk.stem.WordNetLemmatizer()
  stopwords = nltk.corpus.stopwords.words('english')
  text = (unicodedata.normalize('NFKD', text)
    .encode('ascii', 'ignore')
    .decode('utf-8', 'ignore')
    .lower())
  words = re.sub(r'[^\w\s]', '', text).split()
  return [wnl.lemmatize(word) for word in words if word not in stopwords]
words = basic_clean(''.join(str(dfA['result_text'].tolist())))

top_20_unigrams = (pd.Series(nltk.ngrams(words, 1)).value_counts())[:20]
top_20_unigrams.to_excel('top_20_unigrams.xlsx')
print(top_20_unigrams)

top_20_bigrams = (pd.Series(nltk.ngrams(words, 2)).value_counts())[:20]
top_20_bigrams.to_excel('top_20_bigrams.xlsx')
print(top_20_bigrams)

top_20_trigrams = (pd.Series(nltk.ngrams(words, 3)).value_counts())[:20]
top_20_trigrams.to_excel('top_20_trigrams.xlsx')
print(top_20_trigrams)

top_20_quodgrams = (pd.Series(nltk.ngrams(words, 4)).value_counts())[:20]
top_20_quodgrams.to_excel('top_20_quodgrams.xlsx')
print(top_20_quodgrams)

fig = plt.figure(figsize=(12, 8))
top_20_unigrams.sort_values().plot.barh(color='blue', width=.9)
plt.title('20 most frequently occuring unigrams')
plt.ylabel('Unigram')
plt.xlabel('Number of occurances')
fig.tight_layout()
fig.savefig("unigrams.png", dpi=900)
fig.savefig("unigrams.pdf", dpi=900)
plt.show()

fig = plt.figure(figsize=(12, 8))
top_20_bigrams.sort_values().plot.barh(color='blue', width=.9)
plt.title('20 most frequently occuring bigrams')
plt.ylabel('Bigram')
plt.xlabel('Number of occurances')
fig.tight_layout()
fig.savefig("bigrams.png", dpi=900)
fig.savefig("bigrams.pdf", dpi=900)
plt.show()

fig = plt.figure(figsize=(12, 8))
top_20_trigrams.sort_values().plot.barh(color='blue', width=.9)
plt.title('20 most frequently occuring trigrams')
plt.ylabel('Trigram')
plt.xlabel('Number of occurances')
fig.tight_layout()
fig.savefig("trigrams.png", dpi=900)
fig.savefig("trigrams.pdf", dpi=900)
plt.show()

fig = plt.figure(figsize=(12, 8))
top_20_quodgrams.sort_values().plot.barh(color='blue', width=.9)
plt.title('20 most frequently occuring quodgrams')
plt.ylabel('Quodgrams')
plt.xlabel('Number of occurances')
fig.tight_layout()
fig.savefig("quodgrams.png", dpi=900)
fig.savefig("quodgrams.pdf", dpi=900)
plt.show()

print(dfA.head())
#creating age
dfA['age'] = dfA['result_text'].str.findall( r'\d{1,2}(?:yearold)|\d{1,2} (?:year old)|\d{1,2}(?:-year old)|\d{1,2}year old|\d{1,2}(?:-year-old)')
#stripping the list to get actual values
dfA['age'] = dfA['age'].str.get(0)
#To remove the word year
dfA['age'] = dfA['age'].str.findall(r'\d{1,2}')
#stripping the list to get actual values
dfA['age'] = dfA['age'].str.get(0)

fig = plt.figure(figsize=(12, 8))
sns.distplot(dfA.age.astype(float).dropna())
plt.title('age distribution')
fig.tight_layout()
fig.savefig("age_distribution.png", dpi=900)
fig.savefig("age_distribution.pdf", dpi=900)
plt.show()

print('deleting variables not required...')
del dfA['wordCount_before'], dfA['wordCount_after'], dfA['charCount_before'], dfA['charCount_after']
del dfA['uniqueWords_before'], dfA['uniqueWords_after']

dfA['result_text'] = dfA['result_text'].replace('\s{2,}',' ', regex=True)
dfA['result_text'] = dfA['result_text'].replace('major 3/minor 3','major 3, minor 3', regex=True)
dfA['result_text'] = dfA['result_text'].replace('left right lobe','', regex=True)
dfA['result_text'] = dfA['result_text'].replace('major pattern 4/5. minor pattern 5/5','major pattern 4/5 minor pattern 5/5', regex=True)
dfA['result_text'] = dfA['result_text'].replace('\s{2,}',' ', regex=True)
 
#########################################################
#first classification
#########################################################

#Mining the gleason score
dfA['gleason_score'] = dfA['result_text'].str.findall( r'\d \+ \d|\d\+\d|major \d \+ minor \d|major \d\+ minor \d|'
                                                      '\d major \+ \d minor|major \w+ \d\+ minor \w+ \d|'
                                                      'major \w+ \d \+ minor \w+ \d|major\d\+ minor \d|\d major\+ \d minor|'
                                                      '\d major \+ \d minor|\d major \+\d minor|major \d \+ minor compon \d|'
                                                      'major \d \+\d minor|major \d\+ minor score \d')

#stripping the list to get actual values
dfA['gleason_score'] = dfA['gleason_score'].str.get(0)
dfA_subA = dfA[dfA['gleason_score'].notnull()]
dfA_subB = dfA[dfA['gleason_score'].isnull()]

#cleaning the mined text
dfA_subA['gleason_score'] = dfA_subA['gleason_score'].replace('[a-z]+','', regex=True)
dfA_subA['gleason_score'] = dfA_subA['gleason_score'].str.strip()
dfA_subA['gleason_score'] = dfA_subA['gleason_score'].replace('\s+','', regex=True)
dfA_subA[['major','minor']] = dfA_subA.gleason_score.str.split("+", expand=True)
dfA_subA['total'] = dfA_subA['major'].astype(int) + dfA_subA['minor'].astype(int)
dfA_subA['combination'] = dfA_subA['major'].astype(str) + '+' + dfA_subA['minor'].astype(str) + '=' + dfA_subA['total'].astype(str)

#########################################################
#second classification
#########################################################
dfA_subB['gleason_score'] = dfA_subB['result_text'].str.findall( r'\d plu \d|major \d plu minor \d|major \w+ \d minor \w+ \d|'
                                                                'major \d minor \d|\[a-z]+ \[a-z]+ \[a-z]+ \d \[a-z]+ \[a-z]+ \[a-z]+ \d|'
                                                                'gleason major score \d gleason minor score \d|major \d minor compon \d')
#stripping the list to get actual values
dfA_subB['gleason_score'] = dfA_subB['gleason_score'].str.get(0)
dfA_subC = dfA_subB[dfA_subB['gleason_score'].isnull()]
dfA_subB = dfA_subB[dfA_subB['gleason_score'].notnull()]

#cleaning the mind text
dfA_subB['gleason_score'] = dfA_subB['gleason_score'].replace('[a-z]+','', regex=True)
dfA_subB['gleason_score'] = dfA_subB['gleason_score'].str.strip()
dfA_subB['gleason_score'] = dfA_subB['gleason_score'].replace('\s{2,}',' ', regex=True)
dfA_subB[['major','minor']] = dfA_subB.gleason_score.str.split(" ", expand=True)
dfA_subB['total'] = dfA_subB['major'].astype(int) + dfA_subB['minor'].astype(int)
dfA_subB['combination'] = dfA_subB['major'].astype(str) + '+' + dfA_subB['minor'].astype(str) + '=' + dfA_subB['total'].astype(str)

#########################################################
#third classification
#########################################################
dfA_subC['gleason_score'] = dfA_subC['result_text'].str.findall( r'gleason \d,\d|major \d, minor \d|'
                                                                'grade \d,\d|\d major, \d minor|'
                                                                '\[a-z]+ \[a-z]+ \d, \[a-z]+ \[a-z]+ \d|'
                                                                'major pattern \d, minor pattern \d|'
                                                                'gleason score \d,\d')
#stripping the list to get actual values
dfA_subC['gleason_score'] = dfA_subC['gleason_score'].str.get(0)
dfA_subD = dfA_subC[dfA_subC['gleason_score'].isnull()]
dfA_subC = dfA_subC[dfA_subC['gleason_score'].notnull()]

#cleaning the mind text
dfA_subC['gleason_score'] = dfA_subC['gleason_score'].replace('[a-z]+','', regex=True)
dfA_subC['gleason_score'] = dfA_subC['gleason_score'].replace('\s+','', regex=True)
dfA_subC[['major','minor']] = dfA_subC.gleason_score.str.split(",", expand=True)
dfA_subC['total'] = dfA_subC['major'].astype(int) + dfA_subC['minor'].astype(int)
dfA_subC['combination'] = dfA_subC['major'].astype(str) + '+' + dfA_subC['minor'].astype(str) + '=' + dfA_subC['total'].astype(str)

#########################################################
#forth classification
#########################################################
dfA_subD['gleason_score'] = dfA_subD['result_text'].str.findall( r'major \d; minor \d|major\d; minor \d|'
                                                                'major pattern \d; minor pattern \d|'
                                                                '\d major; \d minor')
#stripping the list to get actual values
dfA_subD['gleason_score'] = dfA_subD['gleason_score'].str.get(0)
dfA_subE = dfA_subD[dfA_subD['gleason_score'].isnull()]
dfA_subD = dfA_subD[dfA_subD['gleason_score'].notnull()]

#cleaning the mind text
dfA_subD['gleason_score'] = dfA_subD['gleason_score'].replace('[a-z]+','', regex=True)
dfA_subD['gleason_score'] = dfA_subD['gleason_score'].str.strip()
dfA_subD['gleason_score'] = dfA_subD['gleason_score'].replace('\s+','', regex=True)
dfA_subD[['major','minor']] = dfA_subD.gleason_score.str.split(";", expand=True)
dfA_subD['total'] = dfA_subD['major'].astype(int) + dfA_subD['minor'].astype(int)
dfA_subD['combination'] = dfA_subD['major'].astype(str) + '+' + dfA_subD['minor'].astype(str) + '=' + dfA_subD['total'].astype(str)

#########################################################
#fifth classification
#########################################################
dfA_subE['gleason_score'] = dfA_subE['result_text'].str.findall( r'major pattern \d/\d minor pattern \d/\d')
#stripping the list to get actual values
dfA_subE['gleason_score'] = dfA_subE['gleason_score'].str.get(0)
dfA_subF = dfA_subE[dfA_subE['gleason_score'].isnull()]
dfA_subE = dfA_subE[dfA_subE['gleason_score'].notnull()]

#cleaning the mind text
dfA_subE['gleason_score'] = dfA_subE['gleason_score'].replace('[a-z]+','', regex=True)
dfA_subE['gleason_score'] = dfA_subE['gleason_score'].replace('/5','', regex=True)
dfA_subE['gleason_score'] = dfA_subE['gleason_score'].str.strip()
dfA_subE['gleason_score'] = dfA_subE['gleason_score'].replace('\s{2,}',' ', regex=True)
dfA_subE[['major','minor']] = dfA_subE.gleason_score.str.split(" ", expand=True)
dfA_subE['total'] = dfA_subE['major'].astype(int) + dfA_subE['minor'].astype(int)
dfA_subE['combination'] = dfA_subE['major'].astype(str) + '+' + dfA_subE['minor'].astype(str) + '=' + dfA_subE['total'].astype(str)

#########################################################
#sixth classification
#########################################################
dfA_subF['gleason_score'] = dfA_subF['result_text'].str.findall( r'gleason score \d \d,\d')
#stripping the list to get actual values
dfA_subF['gleason_score'] = dfA_subF['gleason_score'].str.get(0)
dfA_subG = dfA_subF[dfA_subF['gleason_score'].isnull()]
dfA_subF = dfA_subF[dfA_subF['gleason_score'].notnull()]

#cleaning the mind text
dfA_subF['gleason_score'] = dfA_subF['gleason_score'].replace('gleason score 7','', regex=True)
dfA_subF['gleason_score'] = dfA_subF['gleason_score'].str.strip()
dfA_subF[['major','minor']] = dfA_subF.gleason_score.str.split(",", expand=True)
dfA_subF['total'] = dfA_subF['major'].astype(int) + dfA_subF['minor'].astype(int)
dfA_subF['combination'] = dfA_subF['major'].astype(str) + '+' + dfA_subF['minor'].astype(str) + '=' + dfA_subF['total'].astype(str)

#########################################################
#seventh classification
#########################################################
dfA_subG['gleason_score'] = dfA_subG['result_text'].str.findall( r'gleason grade \d|gleason \d')
#stripping the list to get actual values
dfA_subG['gleason_score'] = dfA_subG['gleason_score'].str.get(0)
dfA_subH = dfA_subG[dfA_subG['gleason_score'].isnull()]
dfA_subG = dfA_subG[dfA_subG['gleason_score'].notnull()]

#cleaning the mind text
dfA_subG['gleason_score'] = dfA_subG['gleason_score'].replace('[a-z]+','', regex=True)
dfA_subG['gleason_score'] = dfA_subG['gleason_score'].str.strip()
dfA_subG['major'] = dfA_subG['gleason_score']
dfA_subG['minor'] = dfA_subG['gleason_score']
dfA_subG['total'] = dfA_subG['major'].astype(int) + dfA_subG['minor'].astype(int)
dfA_subG['combination'] = dfA_subG['major'].astype(str) + '+' + dfA_subG['minor'].astype(str) + '=' + dfA_subG['total'].astype(str)

#########################################################
#eighth classification
#########################################################
dfA_subH['gleason_score'] = dfA_subH['result_text'].str.findall( r'gleason score \d major; minor \d')
#stripping the list to get actual values
dfA_subH['gleason_score'] = dfA_subH['gleason_score'].str.get(0)

#cleaning the mind text
dfA_subH['gleason_score'] = dfA_subH['gleason_score'].replace('[a-z]+','', regex=True)
dfA_subH['gleason_score'] = dfA_subH['gleason_score'].str.strip()
dfA_subH['gleason_score'] = dfA_subH['gleason_score'].replace('\s+','', regex=True)
dfA_subH[['major','minor']] = dfA_subH.gleason_score.str.split(";", expand=True)
dfA_subH['major'] = dfA_subH['major'].astype(int) - dfA_subH['minor'].astype(int)
dfA_subH['major'] = dfA_subH['major'].astype(str)
dfA_subH['total'] = dfA_subH['major'].astype(int) + dfA_subH['minor'].astype(int)
dfA_subH['combination'] = dfA_subH['major'].astype(str) + '+' + dfA_subH['minor'].astype(str) + '=' + dfA_subH['total'].astype(str)

#concatenate
dfB = pd.concat([dfA_subA, dfA_subB, dfA_subC, dfA_subD, dfA_subE, dfA_subF, dfA_subG, 
                 dfA_subH], ignore_index=True)

dfB['minor'] = dfB['minor'].str.replace(r'9', '5')
dfB['total'] = dfB['total'].replace(13, 9)
dfB['combination'] = dfB['combination'].replace(r'4+9=13', '4+5=9')

dfB.loc[dfB.total < 7, 'risk'] = "low"
dfB.loc[dfB.total > 7, 'risk'] = "high"
dfB.loc[dfB.total == 7, 'risk'] = "intermediate"

fig = plt.figure(figsize=(12, 8))
dfB['risk'].value_counts().sort_values().plot.barh(color='blue', width=.9)
plt.title('Risk levels')
plt.ylabel('risks')
plt.xlabel('Number of occurances')
fig.tight_layout()
fig.savefig("risks_levels.png", dpi=900)
fig.savefig("risks_levels.pdf", dpi=900)
plt.show()

print('reconnecting to the database')
conn = psycopg2.connect(user="postgres", password="XXXXX",
                                  host="000.0.0.0", port="0000", database="XXXX")

print('deleting table from database if exist')
with conn:
    cursor = conn.cursor()
    cursor.execute('drop table if exists mined_gleason_scores')
    conn.commit()


print('exporting the dataframe to postgres table...')
engine = create_engine('postgresql://postgres:XXXXX@000.0.0.0:0000/XXXX')
dfA.to_sql('mined_gleason_scores', engine, chunksize=100, index=False)
print('data loaded to postgres successfully...') 

print('halting...') 

stoptd = time.strftime('%X %x %Z')
print('\n','The program stop time and Date','\n',stoptd)
print('\n','It took', (time.time()-start)/60, 'minutes to run the script')

print('self clean up...')
del dfA, start, startptd, stoptd, conn, cursor, engine, fig, dfB, porter_stemmer
del eng_stopwords, pattern_to_find, pattern_to_repl, row, ax1, ax2, sql_cmd1
del dfA_subA, dfA_subB, dfA_subC, dfA_subD, dfA_subE, dfA_subF, dfA_subG, dfA_subH
del stopwords, sub_table, text_revi, top_20_bigrams, top_20_quodgrams, top_20_trigrams
del top_20_unigrams, wordcloud, words

print('complete...')  


