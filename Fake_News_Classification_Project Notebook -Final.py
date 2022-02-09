#!/usr/bin/env python
# coding: utf-8

# ## Import the data and packages

# In[1]:


import pandas as pd
df = pd.read_csv('/Users/pamelaafful/Downloads/news.csv')


# In[2]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer
from nltk.corpus import stopwords
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from os import path
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.util import ngrams
get_ipython().system('pip install lime')
get_ipython().system('pip install plotly')
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.offline as pyo
pyo.init_notebook_mode()
from plotly.offline import iplot,init_notebook_mode
init_notebook_mode()
from textblob import TextBlob


# ## The Data
# The data contains three main columns:
# 1. Title- the title of the news tweet.
# 2. Text- the actual text of the news tweet.
# 3. Label- the label of the news tweet either Fake or Real.
# 
# For the purposes of this analysis, I will only focus on the text and label columns

# In[9]:


df.head()


# In[ ]:


len(df) #check length of data


# ### **Preprocessing**
# 
# Firstly, I clean the data by applying the following preprocessing function to the data. The function performs the following:
# 1. First converts all the text to lower case.
# 2. Removes all URL links, nonalphanumeric text, and Twitter handles present in the text.
# 3. Tokenizes the text based on each individual word (or the white space between each word) and puts these in a word list.
# 4. Removes all stop words from the word list. Stop words are common words that appear too frequently in the text. Examples of these include "and", "so", "the" etc. They are necessary for sentence construction (that is, they form the parts of speech of the text) but do not add any additional meaning or value. This proves quite important for the purposes of data storage, model training, and model fine-tuning.
# 5. Then finally converts the "purified" word list into a data column that is then appended to the original data frame as "cleaned_text".

# In[3]:


handles=r'@\w+'
generic_urls=r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'
import re

'''
Takes in text data and performs the following:
lowers the case of text
removes URLs
Removes non alphanumeric text
Tokenizes the text
'''
def preprocess(df_text):
    df_text=df_text.lower()
    df_text=df_text.replace("'","")
    df_text=df_text.replace("“","")
    df_text=df_text.replace("”","")
    df_text=df_text.replace('—','')
    df_text=df_text.replace('’','')
    df_text=df_text.replace('‘','')
    df_text= re.sub(generic_urls, '', df_text) # remove URLs
    re.sub(r'\W+', '', df_text)
    df_text= re.sub(handles, '', df_text) #remove Twitter handles
    df_text = re.sub('<[^>]*>', '', df_text)# remove none alphanumeric text
    df_text = re.sub('<.*?>+', '', df_text)
    df_text = re.sub('[%s]' % re.escape(string.punctuation), '', df_text)
    tokenized_list = nltk.word_tokenize(df_text)
    stop_words=set(stopwords.words('english')) 
    cleaned_list=[i for i in tokenized_list if i not in stop_words]# remove stop words
    return ' '.join(cleaned_list)
df['cleaned_text'] = df['text'].apply(preprocess)
df['cleaned_text'].head()


# ## Exploratory Data Analysis

# In[ ]:


df.head()


# In[32]:


df.groupby('label').count()['text'] #check the data spread between each label


# Data is fairely balanced so accuracy scores will be used as the main metric to evaluate performace.

# ## **Sentiment Analysis**
# 

# Lets take a look at average polarity and subjectivity of each label in the data

# In[4]:


'''
This function takes in a text data and returns the polarity of the text 
Polarity is float which lies in the range of [-1,1] where 1 means positive statement and -1 means a negative statement

'''
def polarity_score(df_text):
  return TextBlob(df_text).sentiment.polarity

'''

This function takes in a text data and returns the subectivity of the text. 
Subjectivity sentences generally refer to personal opinion, emotion or judgment whereas objective refers to factual information. 

Subjectivity is also a float which lies in the range of [0,1].
'''
def subjectivity_score(df_text):
  return TextBlob(df_text).sentiment.subjectivity

df['polarity_score']=df['cleaned_text'].apply(polarity_score) #apply above function to the data
df['subjectivity_score']=df['cleaned_text'].apply(subjectivity_score)#apply above function to the data


# In[ ]:


df.head()


# Lets take a look at the average polarity score and subjectivity score for each data label as well as the overall ploarity and subjectivity for the entire data

# In[ ]:


df['polarity_score'].mean()
df['subjectivity_score'].mean()
print(' The overall polarity of the tweet data is '+ str(round(df['polarity_score'].mean(),2)))
print(' The overall subjectivity of the tweet data is '+ str(round(df['subjectivity_score'].mean(),2)))


# In[ ]:


df2=df[['label','polarity_score','subjectivity_score']]
df2.groupby('label').mean()


# The average sentiment of the data is fairly nuetral for each data label i.e the difference in the average polarities between Fake and Real news is relatively small.The average subjectivity scores are also quite close for each label with the average subjectivity score leaning toward objective news (assuming that 0.5 is the threshold divide between subjective and objective). 

# Plotting the distribution of Sentiment and Polarity Scores

# In[35]:


#Histogram Plot For Overall Polarity Distribution
init_notebook_mode(connected=False)
trace=go.Histogram(x=df.polarity_score)
data=trace
layout={'title':'Overall Polaritry Score Distribution', 'xaxis':{'title':'Polarity Score'},'yaxis':{'title':'Frequency'}}
iplot({'data':data,'layout':layout})


# From the above, it looks like the for the majority of the data (for both labels)the majority of tweets lie on toward the positive end of the spectrum (although still close to Nuetral). 

# In[8]:


#Histogram Plot For Overall Subjectivity Distribution#
init_notebook_mode(connected=False)
trace=go.Histogram(x=df.subjectivity_score)
data=trace
layout={'title':'Overall Subjectivity Score Distribution', 'xaxis':{'title':'Subjectivity Score'},'yaxis':{'title':'Frequency'}}
iplot({'data':data,'layout':layout})


# The subjectivity score is more evenly spread around the mean ( data is slightly subjective). We can take  look at the distribution by label. Most of the tweet data is opinion.

# Lets take a look at the distribution by label:

# In[9]:


fig = make_subplots(rows=2, cols=2,
                    subplot_titles=("Polarity Score Distribution-REAL", "Polarity Score Distribution-FAKE",'Subjectivity Score Distribution-REAL','Subjectivity Score Distribution-FAKE'),
                    x_title="Score",y_title='Frequency')
fig.add_trace(
    go.Histogram(x=df[df['label']=='REAL']['polarity_score']),
    row=1, col=1)#name="Polarity-REAL"
fig.add_trace(
    go.Histogram(x=df[df['label']=='FAKE']['polarity_score']),
    row=1, col=2)#
fig.add_trace(
    go.Histogram(x=df[df['label']=='REAL']['subjectivity_score']),
    row=2, col=1)#name="Subjectivity-REAL",
fig.add_trace(
    go.Histogram(x=df[df['label']=='FAKE']['subjectivity_score']),
    row=2, col=2)


# In[36]:


# Tweet outliers
print('Tweet with the higest polarity:',(df[df['polarity_score']==df['polarity_score'].max()]['cleaned_text'].iloc[0]),'Label of tweet with highest polarity:',df[df['polarity_score']==df['polarity_score'].max()]['label'].iloc[0],sep="\n")

print('Tweet with the lowest polarity:',(df[df['polarity_score']==df['polarity_score'].min()]['cleaned_text'].iloc[0]),'Label of tweet with lowest polarity:',df[df['polarity_score']==df['polarity_score'].min()]['label'].iloc[0],sep="\n")

print('Tweet with the higest subjectivity:',(df[df['subjectivity_score']==df['subjectivity_score'].max()]['cleaned_text'].iloc[0]),'Label of tweet with highest subjectivity:',df[df['subjectivity_score']==df['subjectivity_score'].max()]['label'].iloc[0],sep="\n")

print('Tweet with the lowest subjectivity:',(df[df['subjectivity_score']==df['subjectivity_score'].min()]['cleaned_text'].iloc[0]),'Label of tweet with lowest subjectivity:',df[df['subjectivity_score']==df['subjectivity_score'].min()]['label'].iloc[0],sep="\n")


# Taking a look at word frequency...

# In[5]:


# Before that, we tokenize the text data using the with space between them
def tokenize_text(df_text):
  return df_text.split()
df['tokenized_tweet']=df.cleaned_text.apply(tokenize_text)
df.head()


# In[9]:


word_freq=pd.DataFrame(df['cleaned_text'].str.split(expand=True).stack().value_counts()).reset_index()
word_freq=word_freq.rename(columns={'index':'Word', 0:'Count'})
init_notebook_mode(connected=False)
trace=go.Bar(x=word_freq['Word'][0:20],y=word_freq['Count'][0:20])
data=trace
layout={'title':'Top 20 most Frequent words in across entire tweet data', 'xaxis':{'title':'Word'},'yaxis':{'title':'Count'}}
iplot({'data':data,'layout':layout})


# Looking at top 20 words across each target label

# In[7]:


## Fake News Word Frequency
word_freq_Fake=pd.DataFrame(df[df['label']=='FAKE']['cleaned_text'].str.split(expand=True).stack().value_counts()).reset_index()
word_freq_Fake=word_freq_Fake.drop(10, axis=0)
word_freq_Fake=word_freq_Fake.rename(columns={'index':'Word',0:'Count'})

## Real News Word frequency
word_freq_Real=pd.DataFrame(df[df['label']=='REAL']['cleaned_text'].str.split(expand=True).stack().value_counts()).reset_index()
word_freq_Real=word_freq_Real.rename(columns={'index':'Word',0:'Count'})

fig = make_subplots(rows=1, cols=2,
                    subplot_titles=("Top 20 most frequent words-Fake", "Top 20 most frequent words-Real"),
                    x_title="Word",y_title='Count')
fig.add_trace(
    go.Bar(x=word_freq_Fake['Word'].iloc[0:20], y=word_freq_Fake['Count'].iloc[0:20]),
    row=1, col=1)
fig.add_trace(
    go.Bar(x=word_freq_Real['Word'].iloc[0:20], y=word_freq_Real['Count'].iloc[0:20]),
    row=1, col=2)


# The most frequent word accross the entire data is "said" which is more prominent across the REAL news tweets whilst the most frequent word used across FAKE news is US. The word Trump is also quite prominent for both target labels.

# # Data Modelling

#  
#  **Train Test Split**

# In[15]:


X=df['cleaned_text']
y=df['label']
x_train,x_test,y_train,y_test= train_test_split(X,y, test_size=0.2, random_state=7)

len(x_test)


# **Initialize TDif Vectorizor and applying to training example**
# 

# In[16]:


tdif_vec=TfidfVectorizer(stop_words= 'english',max_df=0.7) 

xtrain_vec=tdif_vec.fit_transform(x_train)
xtest_vec=tdif_vec.transform(x_test)
xtest_vec


# **Using Different Classifiers**
# 
# We first train various models or classifiers on the data to see which model trains the fastest and yields the best accuracy. The classifiers used are Passive-Aggressive, Logistic Regression, Random Forest, SVC, and Multinomial Naive Bayes.

# In[17]:


classifiers=[PassiveAggressiveClassifier(max_iter=50),LogisticRegression(max_iter=150),SVC(),RandomForestClassifier(random_state=0),MultinomialNB()]
predictions=[]
model_time=[]
for i in classifiers:
  start = time.time() 
  Fit=i.fit(xtrain_vec,y_train)
  end = time.time()
  mt=end-start #Measures the different training times for each model
  model_time.append(mt)
  pred=Fit.predict(xtest_vec)
  predictions.append(pred) 


# **Model Evaluation**

# In[18]:


### Accuracy####
accuracy_scores=[]
for i in predictions:
  accuracy_scores.append(accuracy_score(y_test,i))

model_eval = pd.DataFrame(list(zip(accuracy_scores,model_time)), index =['PassiveAggressive', 'LogisticRegression', 'SVC', 'Random Forest','Naive Bayes'],
                                              columns =['Accuracy Score','Train_time(sec)'])
model_eval.sort_values(by='Accuracy Score',ascending=False)


# From the above SVC has the best accuracy score followed by Passive Aggressive Model however SVC take the most time to fit. Despite the accuracy the SVC may be difficult to scale for larger data sets. Let's also take a look at the F1 scores.

# In[18]:


### Classification Report###
from sklearn.metrics import classification_report
models=['PassiveAggressive', 'LogisticRegression', 'SVC', 'Random Forest','Navie Bayes']
for model,i in zip(models,predictions):
 print(model, classification_report(y_test,i,target_names=['FAKE','REAL']), sep="\n")


# Based on the classification report above, SVC and PA classifiers have the best F1-scores although since SVC takes longer to train, PA may be the best predictor to use on newer larger datasets. Also PA a lot more suited for bigger, more transitory data (such as online data from social media) where there is a constant stream of data.

# ## Model Interpretation
# 

# Lets evaluate why the Passive Aggressive model predicts the way it does using LIME. LIME is an interpretability surrogate model which can be used on any black-box model (model-agnostic) and provides interpretability for a single observation prediction (local). For more info on LIME refer to this medium article https://medium.com/@kalia_65609/interpreting-an-nlp-model-with-lime-and-shap-834ccfa124e4. The code below was adapted from the article. We continue with the Passive Aggresive model as I have added .pa_proba method (for converting confidence values into probabilities) so that is readily usuable in the model pipeline.

# In[19]:


# Modified Passive aggresive classifier with predict_proba method
class PA_classifier_prob(PassiveAggressiveClassifier):
  def _init_(self,X,y):
    super()._init_(X,y)
  def predict_proba(self,X): #method to wrap values from decision function in a sigmoid function to get probabilities
    arr1= (1. / (1. + np.exp(-self.decision_function(X)))) 
    arr2= -(arr1-1)
    return np.stack((arr2, arr1), axis=1)


# In[20]:


# Number of non-zero weighted features in the model.
pa = PA_classifier_prob(max_iter=50)
pa.fit(xtrain_vec,y_train)
pa.predict(xtest_vec)
print(str(np.count_nonzero(pa.coef_))+ 
      " out of "+str(xtest_vec.shape[1])+
      ' features were considred relevant by the Passive Aggressive Classifier model.') 


# LIME takes in a pipeline as an input. We first fit the pipeline onto the training data and save this to the variable "model". Since LIME only provides local interpretability I apply LIME to a list of random indices from the x_test vector.

# In[21]:


#Implementing LIME
import lime
import sklearn.ensemble
from __future__ import print_function
from lime import lime_text
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer

# converting the vectoriser and model into a pipeline
# this is necessary as LIME takes a model pipeline as an input
c =Pipeline([('vectorizer', tdif_vec), ('pa', PA_classifier_prob(max_iter=50))])
model=c.fit(x_train,y_train)
# saving a list of strings version of the X_test object
ls_X_test= list(xtest_vec)
class_names = {'FAKE': 'FAKE', 'REAL':'REAL'} #dictionary of class labels


# In[ ]:


#Random Index Generator
import random
indicies=[]
for i in range(4):
  a=random.randrange(0,1267)
  indicies.append(a)
indicies  


# In[23]:


# Applying LIME explainer to a random list of indicies
#indicies=[0, 24, 39, 15]
for idx in indicies:

  LIME_explainer = LimeTextExplainer(class_names=class_names)
  LIME_exp = LIME_explainer.explain_instance(list(x_test)[idx],model.predict_proba)
  # print class names to show what classes the visualization refers to
  print("1 = REAL, 0 = FAKE")
  print('Actual Lable', list(y_test)[idx])
# show the explainability results with highlighted text
  LIME_exp.show_in_notebook(text=True)


# For the indices we have chosen, the visualisations show the following:
# 
# 
# *   The word "said"-which was a the top most frequent word for Real news
# labels- was also heavily weighted by the model 
# 
# *   Interestly, texts containing the year 2016 were weighted in favour for the fake news label prediction. On the other hand, the year 2012 was weighted in favour of the real news label.
# 
# * The results from the fourth example are also quite surprising. The word "cancer" was weighted in favor of the real news label but the word "chemotherapy" was weighted in favor of the fake news label! We would expect these words to move together in favor of a particular label since they are related together but this is not the case for the Passive-Aggressive Classifier.
# 
# 
