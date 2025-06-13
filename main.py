from flask import Flask, request, render_template_string
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import tweepy
import base64
import matplotlib.pyplot as plt
from io import BytesIO
import os
from dotenv import load_dotenv

app = Flask(__name__)

def scrape_twitter_posts(topic, num_posts=50, consumer_key=None, consumer_secret=None,
                         access_token=None, access_token_secret=None):
  if not all([consumer_key, consumer_secret, access_token, access_token_secret]):
    print("Warning: No Twitter/X API credentials provided. Using default posts on LLMs in 2025.")
    posts = [
      "LLMs Are Revolutionizing AI: Here's Why They're the Future of Tech",
      "Gemini's Multimodal Magic: Beating OpenAI at Image and Text?",
      "New LLM Breakthrough: 10x Faster Training, But at What Cost?",
      "Why Open-Source LLMs Are Beating Proprietary Models in 2025",
      "From ChatGPT to Grok: The Evolution of LLMs in Just 3 Years!",
      "Energy Crisis? LLMs Are Eating Up More Power Than Small Countries! #AISustainability",
      "How Gemini Is Powering Google's Search Revolution",
      "How Gemini's Integration with Android Is Changing Mobile AI",
      "OpenAI vs. Google: The LLM Arms Race Heats Up in 2025",
      "DeepSeek's $3M AI Model Challenges $1B U.S. Giants — Efficiency Wins?"
    ]
  else:
    try:
      client = tweepy.Client(
          consumer_key=consumer_key,
          consumer_secret=consumer_secret,
          access_token=access_token,
          access_token_secret=access_token_secret
      )
    except Exception as error:
      print(f"Error initializing Tweepy client: {error}")
      return None
    posts = []
    try:
      tweets = client.search_recent_tweets(query=f"{topic} -is:retweet", max_results=num_posts, tweet_fields=['text'])
      if tweets.data:
        for tweet in tweets.data:
          posts.append(tweet.text)
      else:
        print(f"No posts found for topic: {topic}")
        return []
    except tweepy.TweepyException as error:
      print(f"Error accessing X API: {error}")
      return None
  return posts

def analyze_sentiment(scraped_posts):
  analyzer = SentimentIntensityAnalyzer()
  sentiments = []
  for post in scraped_posts:
    scores = analyzer.polarity_scores(post)
    sentiments.append({
      'Twitter/X Post': post,
      'Negative': scores['neg'],
      'Neutral': scores['neu'],
      'Positive': scores['pos'],
      'Compound': scores['compound']
    })
  return pd.DataFrame(sentiments)

def generate_charts(df):
  plt.figure(figsize=(6, 4))
  plt.bar(['Negative', 'Neutral', 'Positive'], [df['Negative'].mean(), df['Neutral'].mean(), df['Positive'].mean()],
          color=['#ff9999', '#66b3ff', '#99ff99'])
  plt.title('Average Sentiment Scores')
  plt.ylabel('Score')
  bar_io = BytesIO()
  plt.savefig(bar_io, format='png')
  bar_io.seek(0)
  bar_base64 = base64.b64encode(bar_io.getvalue()).decode('utf-8')
  plt.close()
  return bar_base64

HTML_PAGE = '''
<!DOCTYPE html>
<html>
  <head>
    <title>Web Sentiment Analysis</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
      body { font-family: Arial, sans-serif; margin: 40px; }
      .container { max-width: 800px; margin: auto; }
      input[type="text"] { width: 100%; padding: 10px; margin: 10px 0; }
      button { padding: 10px 20px; background-color: #4CAF50; color: white; border: none; cursor: pointer; }
      button:hover { background-color: #45a049; }
      .chart-container { margin: 20px 0; }
      img { max-width: 100%; height: auto; }
    </style>
  </head>
  <body>
    <div class="container">
      <h1>Sentiment Analysis for Twitter/X Posts</h1>
      <form method="POST" action="/">
          <input type="text" name="topic" placeholder="Enter a topic (e.g., electric vehicles)" required>
          <button type="submit">Analyze Sentiment</button>
      </form>
      {% if results %}
      <h2>Results for "{{ topic }}"</h2>
      <p><strong>Average Compound Score:</strong> {{ results.avg_compound|round(3) }}</p>
      <div class="chart-container">
          <h3>Average Sentiment Scores (Bar Chart)</h3>
          <img src="data:image/png;base64,{{ results.bar_chart }}" alt="Bar Chart">
      </div>
      <h3>Twitter/X Posts Analyzed:</h3>
      <ul>
      {% for post in results.twitter_posts %}
          <li>{{ post }}</li>
      {% endfor %}
      </ul>
      {% endif %}
    </div>
  </body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
  results = None
  topic = None
  if request.method == 'POST':
    topic = request.form['topic']
    load_dotenv()
    twitter_posts = scrape_twitter_posts(topic,consumer_key=os.getenv('API_KEY'),
                                          consumer_secret=os.getenv('API_SECRET_KEY'),
                                          access_token=os.getenv('ACCESS_TOKEN'),
                                          access_token_secret=os.getenv('ACCESS_TOKEN_SECRET'))
    df = analyze_sentiment(twitter_posts)
    bar_chart = generate_charts(df)
    results = {
        'avg_compound': df['Compound'].mean(),
        'bar_chart': bar_chart,
        'twitter_posts': twitter_posts
    }
  return render_template_string(HTML_PAGE, results=results, topic=topic)

if __name__ == '__main__':
  app.run(debug=True)
