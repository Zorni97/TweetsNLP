from NLP import *
from argparse import ArgumentParser

parser = ArgumentParser(description="NLP Twitter Analisis")
parser.add_argument("-u", "--usuario", type=str, help="Nombre de usuario Twitter")
args = parser.parse_args()

CONSUMER_KEY = "XXXXXXXXXXXXXXXXXXXXXXXX"
CONSUMER_SECRET = "XXXXXXXXXXXXXXXXXXXXXXXX"
ACCESS_TOKEN = "XXXXXXXXXXXXXXXXXXXXXXXX"
ACCESS_TOKEN_SECRET = "XXXXXXXXXXXXXXXXXXXXXXXX"

if args.usuario is None:
    raise Exception("Argumento -u no encontrado")

tweet_app = NLP(args.usuario, CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
topics = tweet_app.topics()
sentiment = tweet_app.sentiment()
