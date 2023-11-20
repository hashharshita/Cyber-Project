#!/usr/bin/env python
# coding: utf-8

# ###  Importing some useful libraries

# In[1]:


import pandas as pd # use for data manipulation and analysis
import numpy as np # use for multi-dimensional array and matrix

import seaborn as sns # use for high-level interface for drawing attractive and informative statistical graphics 
import matplotlib.pyplot as plt # It provides an object-oriented API for embedding plots into applications
get_ipython().run_line_magic('matplotlib', 'inline')
# It sets the backend of matplotlib to the 'inline' backend:
import time # calculate time 

from sklearn.linear_model import LogisticRegression # algo use to predict good or bad
from sklearn.naive_bayes import MultinomialNB # nlp algo use to predict good or bad

from sklearn.model_selection import train_test_split # spliting the data between feature and target
from sklearn.metrics import classification_report # gives whole report about metrics (e.g, recall,precision,f1_score,c_m)
from sklearn.metrics import confusion_matrix # gives info about actual and predict
from nltk.tokenize import RegexpTokenizer # regexp tokenizers use to split words from text  
from nltk.stem.snowball import SnowballStemmer # stemmes words
from sklearn.feature_extraction.text import CountVectorizer # create sparse matrix of words using regexptokenizes  
from sklearn.pipeline import make_pipeline # use for combining all prerocessors techniuqes and algos

from PIL import Image # getting images in notebook
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator# creates words colud

from bs4 import BeautifulSoup # use for scraping the data from website
from selenium import webdriver # use for automation chrome 
import networkx as nx # for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks.

import pickle# use to dump model 

import warnings # ignores pink warnings 
warnings.filterwarnings('ignore')


# * **Loading the main dataset.**

# In[2]:


phish_data = pd.read_csv('D:\Phishing\phishing_site_urls.csv')


# In[3]:


phish_data.head()


# In[4]:


phish_data.tail()


# In[5]:


phish_data.info()


# * **About dataset**
# * Data is containg 5,49,346 unique entries.
# * There are two columns.
# * Label column is prediction col which has 2 categories 
#     A. Good - which means the urls is not containing malicious stuff and **this site is not a Phishing Site.**
#     B. Bad - which means the urls contains malicious stuffs and **this site is a Phishing Site.**
# * There is no missing value in the dataset.

# In[6]:


phish_data.isnull().sum() # there is no missing values


# * **Since it is classification problems so let's see the classes are balanced or imbalances**

# In[7]:


#create a dataframe of classes counts
label_counts = pd.DataFrame(phish_data.Label.value_counts())
label_counts.head()


# In[8]:


#visualizing target_col
sns.set_style('darkgrid')
sns.barplot(x=label_counts.index, y=label_counts.Label)


# #### RegexpTokenizer
# * A tokenizer that splits a string using a regular expression, which matches either the tokens or the separators between tokens.

# In[9]:


tokenizer = RegexpTokenizer(r'[A-Za-z]+')


# In[10]:


phish_data.URL[0]


# In[11]:


# this will be pull letter which matches to expression
tokenizer.tokenize(phish_data.URL[0]) # using first row


# In[12]:


print('Getting words tokenized ...')

phish_data['text_tokenized'] = phish_data.URL.map(lambda t: tokenizer.tokenize(t)) # doing with all rows


# In[13]:


phish_data.sample(5)


# #### SnowballStemmer
# * Snowball is a small string processing language, gives root words

# In[14]:


stemmer = SnowballStemmer("english") # choose a language


# In[15]:


print('Getting words stemmed ...')
phish_data['text_stemmed'] = phish_data['text_tokenized'].map(lambda l: [stemmer.stem(word) for word in l])


# In[16]:


phish_data.sample(5)


# In[17]:


print('Getting joiningwords ...')
phish_data['text_sent'] = phish_data['text_stemmed'].map(lambda l: ' '.join(l))


# In[18]:


phish_data.sample(5)


# ### Visualization 
# **1. Visualize some important keys using word cloud**

# In[19]:


#sliceing classes
bad_sites = phish_data[phish_data.Label == 'bad']
good_sites = phish_data[phish_data.Label == 'good']


# In[20]:


bad_sites.head()


# In[21]:


good_sites.head()


# * create a function to visualize the important keys from url 

# In[22]:


def plot_wordcloud(text, mask=None, max_words=400, max_font_size=120, figure_size=(24.0,16.0), 
                   title = None, title_size=40, image_color=False):
    stopwords = set(STOPWORDS)
    more_stopwords = {'com','http'}
    stopwords = stopwords.union(more_stopwords)

    wordcloud = WordCloud(background_color='white',
                    stopwords = stopwords,
                    max_words = max_words,
                    max_font_size = max_font_size, 
                    random_state = 42,
                    mask = mask)
    wordcloud.generate(text)
    
    plt.figure(figsize=figure_size)
    if image_color:
        image_colors = ImageColorGenerator(mask);
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");
        plt.title(title, fontdict={'size': title_size,  
                                  'verticalalignment': 'bottom'})
    else:
        plt.imshow(wordcloud);
        plt.title(title, fontdict={'size': title_size, 'color': 'green', 
                                  'verticalalignment': 'bottom'})
    plt.axis('off');
    plt.tight_layout()  


# In[23]:


data = good_sites.text_sent
data.reset_index(drop=True, inplace=True)


# In[24]:


common_text = str(data)
common_mask = np.array(Image.open('D:\Phishing\star.png'))
plot_wordcloud(common_text, common_mask, max_words=400, max_font_size=120, 
               title = 'Most common words use in good urls', title_size=15)


# In[25]:


data = bad_sites.text_sent
data.reset_index(drop=True, inplace=True)


# In[27]:


common_text = str(data)
common_mask = np.array(Image.open('D:\Phishing\comment.png'))
plot_wordcloud(common_text, common_mask, max_words=400, max_font_size=120, 
               title = 'Most common words use in bad urls', title_size=15)


# #### CountVectorizer
# * CountVectorizer is used to transform a corpora of text to a vector of term / token counts.

# In[28]:


#create cv object
cv = CountVectorizer()


# In[29]:


#help(CountVectorizer())


# In[30]:


feature = cv.fit_transform(phish_data.text_sent) #transform all text which we tokenize and stemed


# In[31]:


feature[:5].toarray() # convert sparse matrix into array to print transformed features


# #### * Spliting the data 

# In[32]:


trainX, testX, trainY, testY = train_test_split(feature, phish_data.Label)


# ### LogisticRegression
# 

# In[33]:


# create lr object
lr = LogisticRegression()


# In[34]:


lr.fit(trainX,trainY)


# In[35]:


lr.score(testX,testY)


# In[36]:


print('Training Accuracy :',lr.score(trainX,trainY))
print('Testing Accuracy :',lr.score(testX,testY))
con_mat = pd.DataFrame(confusion_matrix(lr.predict(testX), testY),
            columns = ['Predicted:Bad', 'Predicted:Good'],
            index = ['Actual:Bad', 'Actual:Good'])


print('\nCLASSIFICATION REPORT\n')
print(classification_report(lr.predict(testX), testY,
                            target_names =['Bad','Good']))

print('\nCONFUSION MATRIX')
plt.figure(figsize= (6,4))
sns.heatmap(con_mat, annot = True,fmt='d',cmap="YlGnBu")


# *** So, Logistic Regression is the best fit model, Now we make sklearn pipeline using Logistic Regression**

# In[37]:


pipeline_ls = make_pipeline(CountVectorizer(tokenizer = RegexpTokenizer(r'[A-Za-z]+').tokenize,stop_words='english'), LogisticRegression())


# In[38]:


trainX, testX, trainY, testY = train_test_split(phish_data.URL, phish_data.Label)


# In[39]:


pipeline_ls.fit(trainX,trainY)


# In[40]:


pipeline_ls.score(testX,testY) 


# In[41]:


print('Training Accuracy :',pipeline_ls.score(trainX,trainY))
print('Testing Accuracy :',pipeline_ls.score(testX,testY))
con_mat = pd.DataFrame(confusion_matrix(pipeline_ls.predict(testX), testY),
            columns = ['Predicted:Bad', 'Predicted:Good'],
            index = ['Actual:Bad', 'Actual:Good'])


print('\nCLASSIFICATION REPORT\n')
print(classification_report(pipeline_ls.predict(testX), testY,
                            target_names =['Bad','Good']))

print('\nCONFUSION MATRIX')
plt.figure(figsize= (6,4))
sns.heatmap(con_mat, annot = True,fmt='d',cmap="YlGnBu")


pickle.dump(pipeline_ls,open('phishing.pkl','wb'))



loaded_model = pickle.load(open('phishing.pkl', 'rb'))
result = loaded_model.score(testX,testY)
print(result)


predict_bad = ['yeniik.com.tr/wp-admin/js/login.alibaba.com/login.jsp.php','fazan-pacir.rs/temp/libraries/ipad','tubemoviez.exe','svision-online.de/mgfi/administrator/components/com_babackup/classes/fx29id1.txt']
predict_good = ['youtube.com/','youtube.com/watch?v=qI0TQJI3vdU','retailhellunderground.com/','restorevisioncenters.com/html/technology.html']
loaded_model = pickle.load(open('phishing.pkl', 'rb'))
#predict_bad = vectorizers.transform(predict_bad)
# predict_good = vectorizer.transform(predict_good)
result = loaded_model.predict(predict_bad)
result2 = loaded_model.predict(predict_good)
print(result)
print("*"*30)
print(result2)


predict_bad = ['https://www.geeksforgeeks.org/python-program-to-convert-a-list-to-string/']
loaded_model = pickle.load(open('phishing.pkl', 'rb'))
result = loaded_model.predict(predict_bad)
print(result)

