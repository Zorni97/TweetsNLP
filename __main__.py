from NLP import *
from argparse import ArgumentParser

parser = ArgumentParser(description="NLP Twitter Analisis")
parser.add_argument("-u", "--usuario", type=str, help="Nombre de usuario Twitter")
args = parser.parse_args()

# Establecer tokens de acceso a la API
CONSUMER_KEY = "XXXXXXXXXXXXXXXXXXXXXX"
CONSUMER_SECRET = "XXXXXXXXXXXXXXXXXXXXXX"
ACCESS_TOKEN = "XXXXXXXXXXXXXXXXXXXXXX"
ACCESS_TOKEN_SECRET = "XXXXXXXXXXXXXXXXXXXXXX"

if args.usuario is None:
    raise Exception("Argumento -u no encontrado")

# Crear instancia de la clase principal con los argumentos necesarios
tweet_app = NLP(args.usuario, CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

#Llamar a los métodos de topics y sentiment para extraer el análisis
topics = tweet_app.topics()
sentiment = tweet_app.sentiment()
