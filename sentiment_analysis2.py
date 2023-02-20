from textblob import TextBlob

with open('resources/sentimentanalysis/positive_example.txt') as f:
    text = f.read()

blob = TextBlob(text)
sentiment = blob.sentiment.polarity
print('--------------positive--------------')
print(sentiment)

with open('resources/sentimentanalysis/negative_example.txt') as f:
    text = f.read()

blob = TextBlob(text)
sentiment = blob.sentiment.polarity
print('--------------negative--------------')
print(sentiment)

