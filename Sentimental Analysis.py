#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
get_ipython().system('pip install textblob')
from textblob import TextBlob
get_ipython().system('pip install wordcloud')
from wordcloud import WordCloud
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().system('pip install cufflinks')
import cufflinks as cf
get_ipython().run_line_magic('matplotlib', 'inline')
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True)
cf.go_offline();
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")
warnings.warn("this will not show")
pd.set_option('display.max_columns',None)


# In[8]:


df=pd.read_csv(r"C:\Users\hp\OneDrive\Desktop\amazon.csv")


# In[9]:


df.head()


# In[21]:


# Sort the DataFrame by "wilson_lower_bound" in descending order
df = df.sort_values("wilson_lower_bound", ascending=False)

# Drop the column 'Unnamed:0' from the DataFrame
df.drop(columns=['Unnamed: 0'], inplace=True)

# Display the first few rows of the DataFrame
df.head()


# In[22]:


df.head(5)


# In[25]:


import pandas as pd
import numpy as np

def missing_values_analysis(df):
    na_columns_ = [col for col in df.columns if df[col].isnull().sum() > 0]
    n_miss = df[na_columns_].isnull().sum().sort_values(ascending=True)
    ratio_ = (df[na_columns_].isnull().sum() / df.shape[0] * 100).sort_values(ascending=True)
    missing_df = pd.concat([n_miss, np.round(ratio_, 2)], axis=1, keys=['Missing Values', 'Ratio'])
    missing_df = pd.DataFrame(missing_df)
    return missing_df

def check_dataframe(df, head=5, tail=5):
    print("SHAPE".center(82, '~'))
    print('Rows:{}'.format(df.shape[0]))
    print('Columns:{}'.format(df.shape[1]))
    print("TYPES".center(82, '~'))
    print(df.dtypes)
    print("".center(82, '~'))
    print(missing_values_analysis(df))
    print('DUPLICATED VALUES'.center(83, '~'))
    print(df.duplicated().sum())
    print('QUANTILES'.center(82, '~'))
    print(df.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

# Assuming df is your DataFrame
# Replace df with your DataFrame name
# For example: check_dataframe(df)
# Also, pass the DataFrame to check_dataframe function
check_dataframe(df)


# In[27]:


import pandas as pd

def check_class(dataframe):
    nunique_df = pd.DataFrame({'Variable': dataframe.columns,
                               'Classes': [dataframe[i].nunique() for i in dataframe.columns]})
    nunique_df = nunique_df.sort_values('Classes', ascending=False)
    nunique_df = nunique_df.reset_index(drop=True)
    return nunique_df

# Assuming df is your DataFrame
# Replace df with your DataFrame name
# For example: check_class(df)
# Also, pass the DataFrame to check_class function
check_class(df)


# In[29]:


from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.offline import iplot

constraints=['#B34D22','#EBE00C','#1FEB0C','#0C92EB','#EB0CD5']

def categorical_variable_summary(df, column_name):
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Countplot', 'Percentage'), specs=[[{"type":"xy"}, {'type':'domain'}]])
    
    fig.add_trace(go.Bar(y=df[column_name].value_counts().values.tolist(),
                         x=[str(i) for i in df[column_name].value_counts().index],
                         text=df[column_name].value_counts().values.tolist(),
                         textfont=dict(size=14),
                         name=column_name,
                         textposition='auto',
                         showlegend=False,
                         marker=dict(color=constraints, line=dict(color='#DBE6EC', width=1))),
                  row=1, col=1)
    
    fig.add_trace(go.Pie(labels=df[column_name].value_counts().keys(),
                         values=df[column_name].value_counts().values,
                         textfont=dict(size=18),
                         textposition='auto',
                         showlegend=False,
                         name=column_name,
                         marker=dict(colors=constraints)),
                  row=1, col=2)
    
    fig.update_layout(title={'text': column_name, 'y':0.9, 'x':0.5, 'xanchor':'center', 'yanchor':'top'}, template='plotly_white')
    
    iplot(fig)


# In[30]:


categorical_variable_summary(df,'overall')


# In[31]:


df.reviewText.head()


# In[32]:


review_example=df.reviewText[2031]
review_example


# In[33]:


review_example=re.sub("[^a-zA-Z]",'',review_example)
review_example


# In[43]:


new_list = []
for sublist in review_example:
    new_list.extend(sublist.lower().split())
review_example = new_list
review_example


# In[44]:


rt=lambda x:re.sub("[^a-zA-Z]",' ',str(x))
df["reviewText"]=df["reviewText"].map(rt)
df["reviewText"]=df["reviewText"].str.lower()
df.head()


# In[48]:


# Assuming df is your DataFrame
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import pandas as pd

# Assuming df is your DataFrame
df[['polarity', 'subjectivity']] = df['reviewText'].apply(lambda Text: pd.Series(TextBlob(Text).sentiment))
analyzer = SentimentIntensityAnalyzer()

for index, row in df['reviewText'].iteritems():
    score = analyzer.polarity_scores(row)
    neg = score['neg']
    neu = score['neu']
    pos = score['pos']
    if neg > pos:
        df.loc[index, 'sentiment'] = "Negative"
    elif pos > neg:
        df.loc[index, 'sentiment'] = "Positive"
    else:
        df.loc[index, 'sentiment'] = "Neutral"


# In[49]:


df[df['sentiment']=='Positive'].sort_values("wilson_lower_bound",ascending=False).head(5)


# In[50]:


categorical_variable_summary(df,'sentiment')


# In[ ]:




