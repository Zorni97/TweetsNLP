from NLP import *
from argparse import ArgumentParser

parser = ArgumentParser(description="NLP Twitter Analisis")
parser.add_argument("-u", "--usuario", type=str, help="Nombre de usuario Twitter")
args = parser.parse_args()

CONSUMER_KEY = "KMaDnvMUxzVKBwkgXIvB72Slp"
CONSUMER_SECRET = "FoeYigTb06lQhoXqeIhiIGKanfw4t69ay101ZxMHsMUdrpvlV9"
ACCESS_TOKEN = "1072524873220702208-9O6qIlnUZAG9JSJrBgqSLPQnv1bmwI"
ACCESS_TOKEN_SECRET = "OOCHaSE9aV47TMGiVBEz5vGuMs3WP6cqlzdZzVu804YHq"

if args.usuario is None:
    raise Exception("Argumento -u no encontrado")

tweet_app = NLP(args.usuario, CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
topics = tweet_app.topics()
sentiment = tweet_app.sentiment()
