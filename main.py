import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

analyzer = SentimentIntensityAnalyzer()
sentiments = []
headlines = [
  "Los Angeles ICE detention protests",
  "Elon Musk deletes his X posts attacking Donald Trump",
  "Apple's WWDC reveals a new 'Liquid Glass' aesthetic",
  "China and US reach a trade deal to Trump's liking"
]
for headline in headlines:
  scores = analyzer.polarity_scores(headline)
  sentiments.append({
    'Headline': headline,
    'Negative': scores['neg'],
    'Neutral': scores['neu'],
    'Positive': scores['pos'],
    'Compound': scores['compound']
  })

for sentiment in sentiments:
  plt.figure(figsize=(8, 5))
  sns.barplot(x=['Negative', 'Neutral', 'Positive'], y=[sentiment['Negative'], sentiment['Neutral'], sentiment['Positive']])
  plt.suptitle(f'Sentiment Scores for {sentiment['Headline']}')
  plt.title(f"Compound Sentiment: {sentiment['Compound']}")
  plt.ylabel('Score')
  plt.savefig(f'output/sentiment_{sentiment['Headline'].replace(" ", "_")}.png')
  plt.close()
