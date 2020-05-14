from nltk import word_tokenize
from nltk.corpus import stopwords
import tweepy  # https://github.com/tweepy/tweepy
import re
from datetime import datetime, timedelta
import unicodedata
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import time
import pickle
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


class NLP(object):
    #Se establece el momento del tiempo 24h atrás para buscar tweets con fecha posterior a ese momento
    __now = datetime.now()
    __prev = __now - timedelta(days=1)
    __now = __now.strptime(__now.strftime("%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H:%M:%S")
    __prev = __prev.strptime(__prev.strftime("%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H:%M:%S")

    def __init__(self, screen_name, consumer_key, consumer_secret, access_token, access_token_secret):

        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.access_key = access_token
        self.access_secret = access_token_secret
        self.screen_name = screen_name

        try:
            self.__auth = tweepy.OAuthHandler(self.consumer_key, self.consumer_secret)
            self.__auth.set_access_token(self.access_key, self.access_secret)
            self.__api = tweepy.API(self.__auth)
        except:
            print("Error: Authentication Failed")

        self.tweets = None
        self.tweets_with_sentiment = []
        # Importamos el modelo, el vectorizador y el dataframe con los topics
        self.vectorizer = pickle.load(open('./files/tf_vectorizer.pkl', 'rb'))
        self.model = pickle.load(open('./files/lda.pkl', 'rb'))
        self.df_topic_keywords = pickle.load(open('./files/df_topic_keywords.pkl', 'rb'))
        self.stop_words = stopwords.words()

    #método que devuelve el análisis de topics
    def topics(self, documents=None, vector=None):

        if documents is not None:
            self.tweets = documents
        elif self.tweets is not None:
            pass
        else:
            self.tweets = self.__new_tweets()

        if vector is not None:
            tf = vector
        else:
            tf = self.__vector(self.tweets)

        topic_probability_scores = self.model.transform(tf)
        all_topics = [self.df_topic_keywords.iloc[np.argmax(entry), -1] for entry in topic_probability_scores]
        topics = Counter(all_topics)
        
        plt.pie([float(v) for v in topics.values()], labels=[k for k in topics.keys()],
                autopct='%1.1f%%', startangle=90)
        plt.title(f"Temas de @{self.screen_name}") 
        plt.show()

    #método que devuelve el análisis de sentimiento
    def sentiment(self, documents=None):

        if documents is not None:
            self.tweets = documents
        elif self.tweets is not None:
            pass
        else:
            self.tweets = self.__new_tweets()

        if len(self.tweets_with_sentiment) == 0:
            self.__tweet_dictionary()
        else:
            pass

        pos_tweets = [tweet for tweet in self.tweets_with_sentiment if tweet['sentiment'] == 'positive']
        # Porcentaje tweets positivos
        print("\nPorcentaje tweets positivos: {} %\n".format(
            round(100 * len(pos_tweets) / len(self.tweets_with_sentiment), 2)))
        # picking negative tweets from tweets
        neg_tweets = [tweet for tweet in self.tweets_with_sentiment if tweet['sentiment'] == 'negative']
        # Porcentaje tweets negativos
        print("Porcentaje tweets negativos: {} %\n".format(
            round(100 * len(neg_tweets) / len(self.tweets_with_sentiment), 2)))
        # Porcentaje tweets neutrales
        print("Porcentaje tweets neutrales: {} % \
              \n".format(round(
            100 * (len(self.tweets_with_sentiment) - len(neg_tweets) - len(pos_tweets)) / len(
                self.tweets_with_sentiment), 2)))

    @staticmethod  # método estático que devuelve el sentimiento de los documentos/tweets que le pases
    def get_tweet_sentiment(tweet):
        
        # Utiliza Textblob para analizar el sentimiento
        analysis = TextBlob(tweet)
        # Categorizar el sentimiento y devolverlo
        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity == 0:
            return 'neutral'
        else:
            return 'negative'

    # En el caso de ser llamado, este método devuelve los tweets de las últimas 24 horas para el usuario introducido
    def __new_tweets(self):

        all_tweets = []
        try:
            new_tweets = self.__api.user_timeline(screen_name=self.screen_name,
                                                  count=200, tweet_mode="extended")
        except tweepy.TweepError as e:
            # print error (si hay algún problema con tweepy)
            print("Error : " + str(e))

        print(f"Recibiendo Tweets de @{self.screen_name} de las últimas 24h:\n")

        for tweet in new_tweets:
            if tweet.created_at > self.__prev:
                text = unicodedata.normalize('NFKD', tweet.full_text)
                if len(text) > 0:
                    try:
                        clean_text = ' '.join(
                            re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\ / \ / \S+)", " ", text).split())
                        all_tweets.append(clean_text)
                    except:
                        continue
            else:
                break

        for i in tqdm(range(len(all_tweets))):
            time.sleep(0.009)

        return all_tweets

    # Este método crea una lista de diccionarios por cada tweet, con su texto y sentimiento como keys
    def __tweet_dictionary(self):

        if len(self.tweets_with_sentiment) > 0:
            self.tweets_with_sentiment = []
        else:
            pass

        for tweet in self.tweets:
            # se crea un diccionario por cada tweet
            tweet_sentiment = {'text': tweet, 'sentiment': self.get_tweet_sentiment(tweet)}
            # se hace un append a la lista de tweets con sentimiento
            self.tweets_with_sentiment.append(tweet_sentiment)

    # Devuelve un vector en función del vectorizador que se importe, teniendo como base los documentos lematizados
    def __vector(self, documents):

        lemmatized_documents = []
        document_cleaning = list(
            map(lambda x: re.sub('[\'\"“”’$€\-@:—;\(\),\.!?\d]', '', x), documents))
        document_cleaning = list(map(lambda x: x.lower(), document_cleaning))
        tokenized_documents = [word_tokenize(i) for i in document_cleaning]

        print("\nLematizando Tweets:\n")
        for i in tqdm(range(len(tokenized_documents))):
            lemmatized_documents.append(list())
            for word in tokenized_documents[i]:
                if word not in self.stop_words:
                    try:
                        lemmatized_documents[i].append(WordNetLemmatizer.lemmatize(word))
                    except:
                        lemmatized_documents[i].append(word)
            time.sleep(0.009)
        vector = self.vectorizer.transform(lemmatized_documents)
        print("\n")

        return vector
