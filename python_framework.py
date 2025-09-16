# Part 1: Data Loading and Basic Exploration
import pandas as pd

# Load dataset (replace path with your actual metadata.csv file)
df =pd.read_csv("metadata.csv")

# Examine first few rows
print("First 5 rows:")
print(df.head())

# Dimensions
print("\nShape of dataset:", df.shape)

# Data types
print("\nData types:")
print(df.dtypes)

# Missing values
print("\nMissing values:")
print(df.isnull().sum())

# Basic statistics
print("\nBasic statistics (numeric columns):")
print(df.describe())



#Part 2
# Handle missing data
# Keep only relevant columns
df = df[['title', 'abstract', 'publish_time', 'journal', 'source_x']]

# Drop rows with missing titles or abstracts
df = df.dropna(subset=['title', 'abstract'])

# Convert publish_time to datetime
df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')

# Extract year
df['year'] = df['publish_time'].dt.year

# Add abstract word count
df['abstract_word_count'] = df['abstract'].apply(lambda x: len(str(x).split()))

print("\nCleaned dataset shape:", df.shape)


#part 3
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Papers per year
papers_per_year = df['year'].value_counts().sort_index()
plt.figure(figsize=(8,6))
sns.barplot(x=papers_per_year.index, y=papers_per_year.values)
plt.title("COVID-19 Research Papers by Year")
plt.xlabel("Year")
plt.ylabel("Number of Papers")
plt.show()

# Top journals
top_journals = df['journal'].value_counts().head(10)
plt.figure(figsize=(10,6))
sns.barplot(y=top_journals.index, x=top_journals.values)
plt.title("Top 10 Journals Publishing COVID-19 Research")
plt.xlabel("Number of Papers")
plt.ylabel("Journal")
plt.show()

# Word Cloud of Titles
text = " ".join(df['title'].dropna().astype(str).tolist())
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Most Frequent Words in Paper Titles")
plt.show()


#part 4( streamlit app)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("metadata.csv")
    df = df[['title','abstract','publish_time','journal','source_x']].dropna(subset=['title','abstract'])
    df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')
    df['year'] = df['publish_time'].dt.year
    return df

df = load_data()

st.title("CORD-19 Data Explorer")
st.write("Simple exploration of COVID-19 research papers")

# Interactive filter
years = st.slider("Select Year Range", int(df['year'].min()), int(df['year'].max()), (2020,2021))
filtered = df[(df['year'] >= years[0]) & (df['year'] <= years[1])]

# Papers per year
st.subheader("Papers Published per Year")
papers_per_year = filtered['year'].value_counts().sort_index()
st.bar_chart(papers_per_year)

# Top journals
st.subheader("Top Journals")
top_journals = filtered['journal'].value_counts().head(10)
st.bar_chart(top_journals)

# Wordcloud
st.subheader("Word Cloud of Titles")
text = " ".join(filtered['title'].dropna().astype(str).tolist())
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
st.image(wordcloud.to_array())
