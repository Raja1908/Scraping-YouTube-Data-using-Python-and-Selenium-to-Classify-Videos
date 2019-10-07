#!/usr/bin/env python
# coding: utf-8

# In[11]:


from selenium import webdriver 
import pandas as pd 
from selenium.webdriver.common.by import By 
from selenium.webdriver.support.ui import WebDriverWait 
from selenium.webdriver.support import expected_conditions as EC
driver = webdriver.Chrome('C:\Program Files (x86)\Google\Chrome\chromedriver.exe') 
driver.get("https://www.youtube.com/results?search_query=travel&sp=EgIQAQ%253D%253D")


# In[17]:


user_data = driver.find_elements_by_xpath('//*[@id="video-title"]')
links = []
for i in user_data:
            links.append(i.get_attribute('href'))
print(len(links))


# In[18]:


df = pd.DataFrame(columns = ['link', 'title', 'description', 'category'])


# In[19]:


wait = WebDriverWait(driver, 10)
v_category = "Travel"
for x in links:
            driver.get(x)
            v_id = x.strip('https://www.youtube.com/watch?v=')
            v_title = wait.until(EC.presence_of_element_located(
                           (By.CSS_SELECTOR,"h1.title yt-formatted-string"))).text
            v_description =  wait.until(EC.presence_of_element_located(
                                         (By.CSS_SELECTOR,"div#description yt-formatted-string"))).text
            df.loc[len(df)] = [v_id, v_title, v_description, v_category]


# In[20]:


df


# In[70]:


driver = webdriver.Chrome('C:\Program Files (x86)\Google\Chrome\chromedriver.exe')
driver.get("https://www.youtube.com/results?search_query=science&sp=EgIQAQ%253D%253D")


# In[79]:


import time
elem = driver.find_element_by_tag_name("body")

no_of_pagedowns = 300

while no_of_pagedowns:
    elem.send_keys(Keys.PAGE_DOWN)
    user_data1 = driver.find_elements_by_xpath('//*[@id="video-title"]')
    time.sleep(0.2)
    no_of_pagedowns-=1
links1 = []
for i in user_data1:
            links1.append(i.get_attribute('href'))
print(len(links1))


# In[73]:


df_science = pd.DataFrame(columns = ['link', 'title', 'description', 'category'])


# In[ ]:





# In[74]:


del links1[0]
wait = WebDriverWait(driver, 10)
v_category1 = "Science"
for x in links1:
            driver.get(x)
            v_id = x.strip('https://www.youtube.com/watch?v=')
            v_title = wait.until(EC.presence_of_element_located(
                           (By.CSS_SELECTOR,"h1.title yt-formatted-string"))).text
            v_description =  wait.until(EC.presence_of_element_located(
                                         (By.CSS_SELECTOR,"div#description yt-formatted-string"))).text
            df_science.loc[len(df_science)] = [v_id, v_title, v_description, v_category1]


# In[75]:


driver = webdriver.Chrome('C:\Program Files (x86)\Google\Chrome\chromedriver.exe')
driver.get("https://www.youtube.com/results?search_query=Art+%26+Dance&sp=EgIQAQ%253D%253D")


# In[80]:


import time
elem = driver.find_element_by_tag_name("body")

no_of_pagedowns = 190

while no_of_pagedowns:
    elem.send_keys(Keys.PAGE_DOWN)
    user_data3 = driver.find_elements_by_xpath('//*[@id="video-title"]')
    time.sleep(0.2)
    no_of_pagedowns-=1
links3 = []
for i in user_data1:
            links3.append(i.get_attribute('href'))
print(len(links3))


# In[ ]:


df_science


# In[81]:


df_art = pd.DataFrame(columns = ['link', 'title', 'description', 'category'])


# In[82]:


wait = WebDriverWait(driver, 10)
v_category4 = "Art & Dance"
for x in links1:
            driver.get(x)
            v_id = x.strip('https://www.youtube.com/watch?v=')
            v_title = wait.until(EC.presence_of_element_located(
                           (By.CSS_SELECTOR,"h1.title yt-formatted-string"))).text
            v_description =  wait.until(EC.presence_of_element_located(
                                         (By.CSS_SELECTOR,"div#description yt-formatted-string"))).text
            df_art.loc[len(df_art)] = [v_id, v_title, v_description, v_category4]


# In[83]:


df_art


# In[84]:


frames = [df, df_science,df_art]
df_copy = pd.concat(frames, axis=0, join='outer', join_axes=None, ignore_index=True,
                            keys=None, levels=None, names=None, verify_integrity=False, copy=True)


# In[85]:


df_copy.to_csv(file_name, sep='\t')


# In[86]:


df.to_csv(out.csv, sep='\t')


# In[89]:


df_copy.to_csv(r'M:\file3.csv', index=False)


# In[90]:


import re 
import nltk 
nltk.download('stopwords') 
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer


# In[128]:


df_link = pd.DataFrame(columns = ["link"])        
df_title = pd.DataFrame(columns = ["title"])        
df_description = pd.DataFrame(columns = ["description"])        
df_category = pd.DataFrame(columns = ["category"])        
df_link = df_copy['link'] 
df_title['title'] = df_copy['title'] 
df_description['description'] = df_copy['description'] 
df_category['category'] = df_copy['category']


# In[92]:


len(df_copy.index)


# In[96]:


corpus = []        
for i in range(0, 1553):
    review = re.sub('[^a-zA-Z]', ' ', df_title[i])            
    review = review.lower()            
    review = review.split()            
    ps = PorterStemmer()            
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]            
    review = ' '.join(review)            
    corpus.append(review)


# In[98]:


corpus1 = [] 
for i in range(0, 1553):            
    review = re.sub('[^a-zA-Z]', ' ', df_description[i])            
    review = review.lower()            
    review = review.split()            
    ps = PorterStemmer()            
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]            
    review = ' '.join(review)            
    corpus1.append(review)


# In[122]:


dftitle = pd.DataFrame({'title':corpus})
dfdescription = pd.DataFrame({'description':corpus1})


# In[132]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
dfcategory1 = df_category.apply(LabelEncoder().fit_transform)


# In[133]:


df_new = pd.concat([df_link, dftitle, dfdescription, dfcategory1], axis=1, join_axes = [df_link.index])


# In[115]:


from sklearn.feature_extraction.text import CountVectorizer   
cv = CountVectorizer(max_features = 1500) 
X = cv.fit_transform(corpus, corpus1).toarray() 
y = df_new.iloc[:, 3].values


# In[135]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# In[137]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy')
classifier.fit(X_train, y_train)


# In[138]:


y_pred = classifier.predict(X_test)
classifier.score(X_test, y_test)


# In[139]:


print(classification_report(y_test, y_pred))


# In[141]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import *
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn.metrics import *
print(classification_report(y_test, y_pred))


# In[142]:


from sklearn.svm import SVC
classifier1 = SVC(kernel = 'linear', random_state = 0)
classifier1.fit(X_train, y_train)


# In[143]:


y_pred1 = classifier1.predict(X_test)
classifier1.score(X_test, y_test)


# In[145]:


from xgboost import XGBClassifier
classifier3 = XGBClassifier()
classifier3.fit(X_train, y_train)


# In[146]:


y_pred3 = classifier3.predict(X_test)
classifier3.score(X_test, y_test)


# In[ ]:




