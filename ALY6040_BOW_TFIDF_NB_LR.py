#!/usr/bin/env python
# coding: utf-8

# In[33]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from random import sample
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import numpy as np
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.externals import joblib
import pickle
import lime
import lime.lime_tabular


# In[23]:


reviews = pd.read_csv("amazon_reviews.csv")
reviews.head()


# In[4]:


reviews.shape


# In[5]:


pd.set_option('display.max_colwidth',1000)
reviews[['Summary','Text']].head()


# #### Checking for missing values

# In[7]:


reviews[['Summary', 'Text']].isnull().sum()


# We see that there are 27 missing values in "Summary" and no missing values in "Text". So, we are good to go

# In[8]:


fig, ax = plt.subplots()
sns.distplot(reviews['Score'], ax=ax, kde=False, color='r')

def change_width(ax, new_value) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)

change_width(ax, .4)
plt.xlabel('Score')
plt.ylabel('Count of Reviews')
plt.title('Histogram of Review Scores')
plt.show()


# In[9]:


reviews['Score'].value_counts()/reviews['Score'].count()*100


# We see that ~78% of the scores are 4&5. The remaining 22% belong to 1,2&3

# In[10]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Define a function to plot word cloud
def plot_cloud(wordcloud):
    # Set figure size
    plt.figure(figsize=(40, 30))
    # Display image
    plt.imshow(wordcloud) 
    # No axis details
    plt.axis("off");


# In[12]:


text = reviews['Summary'][1:10000].to_string()
mask = np.array(Image.open('upvote.png'))

# Generate wordcloud
wordcloud = WordCloud(width=3000, height=2000, random_state=1, background_color='white', colormap='Set2', collocations=False, stopwords=STOPWORDS, mask=mask).generate(text)

# Plot
plot_cloud(wordcloud)


# In[54]:


text = reviews['Text'][1:10000].to_string()
mask = np.array(Image.open('comment.png'))

# Generate wordcloud
wordcloud = WordCloud(width=3000, height=2000, random_state=1, background_color='white', colormap='Set2', collocations=False, stopwords=STOPWORDS, mask=mask).generate(text)

# Plot
plot_cloud(wordcloud)


# **Converting Scores 1,2,3 to 0 and 4,5 to 1 to convert the problem statement to binary classification. Removing '3' reviews as they are neutral reviews and not considering them**

# In[1]:


reviews = pd.read_csv("amazon_reviews.csv")
reviews = reviews[reviews['Score'] != 3]
reviews['Score'].replace([1,2], 0, inplace=True)
reviews['Score'].replace([4,5], 1, inplace=True)
reviews['Score'].value_counts()/reviews['Score'].count()*100


# Removing duplicate values

# In[2]:


reviews = reviews.drop_duplicates(subset={"UserId","ProfileName","Time","Text"})
reviews.shape


# **Also given that, Helpfulness Numerator should be less than or equal to Helpfulness Denominator. This means that atleast 1 person has said that this review is helpful. Hence filtering out those values where it is opposite**

# In[3]:


reviews = reviews[reviews['HelpfulnessNumerator'] <= reviews['HelpfulnessDenominator']]
reviews.shape


# ### Combing "Summary" and "Text" which will serve as the complete text for our modeling

# In[4]:


reviews['Text_Clean'] = reviews['Summary'].str.cat(reviews['Text'], sep =". ")
reviews[['Summary', 'Text', 'Text_Clean']].head()


# The Text_Clean is now a combination of both Summary and Text

# In[5]:


reviews = reviews[['Score', 'Text_Clean']]
reviews = reviews.drop([33958])
reviews.reset_index(inplace=True)


# Our data set is ready for modeling purpose

# ## Data Cleaning

# **Stemming - Stopwords**

# In[120]:


nltk.download('stopwords')


# In[122]:


start_time = time.monotonic()
print(time.ctime())

ps = PorterStemmer()
corpus = []
for i in range(0, len(reviews)):
    review = re.sub('[^a-zA-Z]', ' ', reviews['Text_Clean'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    if i % 10000 == 0:
        print('Performing record: ', i)
    
print('minutes: ',(time.monotonic() - start_time)/60)


# In[126]:


reviews['Text_STEM_CLEAN'] = corpus
reviews


# In[128]:


reviews.to_csv("Amazon_Reviews_New.csv")


# ## Bag of Words

# In[36]:


pd.set_option('display.max_colwidth',1000)
reviews = pd.read_csv("Amazon_Reviews_New.csv")
reviews = reviews[['Score', 'Text_STEM_CLEAN']]
reviews.head()


# ### Sampling the data as the system cannot handle such huge data

# Taking 30,000 random observations from the dataset 

# In[40]:


reviews_sample = reviews.sample(n=30000, replace=False, random_state=42)
reviews_sample.shape


# In[41]:


## Applying Countvectorizer
# Creating the Bag of Words model
start_time = time.monotonic()
print(time.ctime())

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,ngram_range=(1,3))
X = cv.fit_transform(reviews_sample['Text_STEM_CLEAN']).toarray()

print('minutes: ',(time.monotonic() - start_time)/60)


# In[42]:


X.shape


# ## TF-IDF

# For TF-IDF let's use 100,000 reviews as our dataset.

# In[43]:


reviews_sample = reviews.sample(n=100000, replace=False, random_state=42)
reviews_sample.shape


# In[21]:


## TFidf Vectorizer
tfidf_v=TfidfVectorizer(max_features=5000,ngram_range=(1,3))

start_time = time.monotonic()
print(time.ctime())
X=tfidf_v.fit_transform(reviews_sample['Text_STEM_CLEAN']).toarray()
print('minutes: ',(time.monotonic() - start_time)/60)


# In[22]:

pickle.dump(tfidf_v, open('transform.pkl', 'wb'))

X.shape


# In[23]:


y=reviews_sample['Score']


# In[24]:


## Divide the dataset into Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[25]:


count_df = pd.DataFrame(X_train, columns=tfidf_v.get_feature_names())
count_df.head()


# In[26]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    See full source and example: 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# ## Multinomial Naive Bayes Classifier

# In[28]:


classifier=MultinomialNB()
classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
f1_score = metrics.f1_score(y_test, pred)
precision = metrics.precision_score(y_test, pred)
recall = metrics.recall_score(y_test, pred)
print("accuracy: %0.3f" % score)
print("precision: %0.3f" % precision)
print("recall: %0.3f" % recall)
print("f1_score: %0.3f" % f1_score)

cm = metrics.confusion_matrix(y_test, pred)
plot_confusion_matrix(cm, classes=['NEGATIVE', 'POSITIVE'])


# **In theory, Multinomial Naive Bayes works well with text data as it works on probability. We can see that overall model is doing well but not doing very well in False Positives** 

# That is because there is a Class Imbalance Issue. ~85% reviews are positive while only ~15% are negative. We have to fix the issue. There are several ways in fixing that particular issue. 
# 
# 1. Undersampling Techniques
# 2. Oversampling Techniques - SMOTE
# 3. Combination of the above both - SMOTETomek and SMOTEENN
# 4. One Class SVM
# 5. Use weights in ML Algorithms (Logistic Regression, Tree models, Bagging and Boosting Algorithmns)
# 6. Weighted Neural Networks

# The dataset is complex enough. SMOTETomek and SMOTEENN increases the dimensions of the data. The implementation of ML part will be computationally expensive. Hence, we will be using **class weights** in ML Algorithmns. We first use Weighted Logistic Regression

# ## Logistic Regression with Weights to tackle Imbalance

# In[31]:


classifier=LogisticRegression(class_weight='Balanced', max_iter=200)
classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
f1_score = metrics.f1_score(y_test, pred)
precision = metrics.precision_score(y_test, pred)
recall = metrics.recall_score(y_test, pred)
print("accuracy: %0.3f" % score)
print("precision: %0.3f" % precision)
print("recall: %0.3f" % recall)
print("f1_score: %0.3f" % f1_score)

cm = metrics.confusion_matrix(y_test, pred)
plot_confusion_matrix(cm, classes=['NEGATIVE', 'POSITIVE'])



# ### We can see that the False Positives has decreased and the F1_score has increased from the before algorithm

# ##  Local Interpretable Model-agnostic Explanations (LIME)

# In[ ]:


predict_fn_logreg = lambda x: classifier.predict_proba(x).astype(float)

explainer = lime.lime_tabular.LimeTabularExplainer(X_train)


# In[ ]:

filename = 'review_score_classifier.pkl'
pickle.dump(classifier, open(filename, 'wb'))



