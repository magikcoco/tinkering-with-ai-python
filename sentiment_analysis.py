from textblob import TextBlob
from newspaper import Article

#import nltk
#nltk.download('punkt')

# get the url
url = 'https://en.wikipedia.org/wiki/Mathematics'
# create an article object from the url
article = Article(url)
# download the article
article.download()
# fully parse the article text
article.parse()
# prepare the article text for natural language processing
article.nlp()

text = article.summary
blob = TextBlob(text)
sentiment = blob.sentiment.polarity  # -1 to 1
print(text)
print(sentiment)
print('\n')

# get the url
url = 'https://www.cnbc.com/2023/01/12/stock-market-futures-open-to-close-news.html'
# create an article object from the url
article = Article(url)
# download the article
article.download()
# fully parse the article text
article.parse()
# prepare the article text for natural language processing
article.nlp()

text = article.summary
blob = TextBlob(text)
sentiment = blob.sentiment.polarity  # -1 to 1
print(text)
print(sentiment)
print('\n')

# get the url
url = 'https://www.cnbc.com/2020/04/22/recession-depth-will-be-much-worse-than-2007-2009-lakshman-achuthan.html'
# create an article object from the url
article = Article(url)
# download the article
article.download()
# fully parse the article text
article.parse()
# prepare the article text for natural language processing
article.nlp()

text = article.summary
blob = TextBlob(text)
sentiment = blob.sentiment.polarity  # -1 to 1
print(text)
print(sentiment)
print('\n')

# get the url
url = 'https://abc7ny.com/four-dead-fatal-shooting-juvenile-killed-linden/12843627/'
# create an article object from the url
article = Article(url)
# download the article
article.download()
# fully parse the article text
article.parse()
# prepare the article text for natural language processing
article.nlp()

text = article.summary
blob = TextBlob(text)
sentiment = blob.sentiment.polarity  # -1 to 1
print(text)
print(sentiment)
print('\n')

