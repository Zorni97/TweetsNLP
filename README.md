# Tweets NLP
Se trata de una práctica de Procesamiento de Lenguaje natural para el máster, en la que creo un pequeño programa de análisis de sentimientos y de temas de tweets con la ayuda de la API de Twitter. El objetivo de esta práctica es poder analizar los temas de los que habla un usuario en sus tweets de las últimas 24 horas, y además llevar a cabo un simple análisis del sentimiento de los mismos. Como el modelo se ha entranado con un dataset en inglés, solo se podrá hacer este análisis a tweets escritos en inglés. 
## Archivos
1. **requirements.txt** donde se encuentras los paquetes necesarios de Python
1. **main.py** se encarga de ejecutar el programa, y es donde hay que introducir los token de la API de twitter
1. **NLP.py** donde se encuentra la clase principal de nuestro programa
1. **TopicModeling.py** es donde se lleva a cabo el entrenamiento de los modelos para el análisis de temas
1. **AnalisisModelos.ipynb** es donde se lleva a cabo el análisis de resultados de los modelos LDA y NMF, además de hacer el GridSearch para LDA y crear los topics con los que clasifica NLP
1. **files** es la carpeta donde se encuentran todos los archivos necesarios para correr el programa. Se encuentra ahi el dataset con el que se entrena el modelo, asi como los archivos **pickle** de los modelos entrenados y de los topics.
## Ejecutar Programa
1. Se necesita en primer lugar tener instaladas las dependencias de paquetes Python con los que trabaja el programa, por lo que en el fichero  **requirements.txt**  vendrán todos los paquetes necesarios, y para instalarlos bastará con ejecutar el comando `pip install -r requirements.txt` que comprobará si ya están instalados o falta alguno.
1. En segundo lugar para poder correr el programa se necesitará introducir las claves de autenticación de la API de Twitter en el archivo **main** .

> CONSUMER_KEY = "XXXXXXXXXXXXXXXXXXXX"
> CONSUMER_SECRET = "XXXXXXXXXXXXXXXXX"
> ACCESS_TOKEN = "XXXXXXXXXXXXXXXXX"
> ACCESS_TOKEN_SECRET = "XXXXXXXXXXXXXXXXXXXX"

1. Y por último para poder ejecutar el programa es necesario introducir por línea de comandos el argumento **-u** seguido del nombre de usuario del que se desea hacer el análisis. Por ejemplo:
`python . -u elonmusk`

* ## Análisis de Temas
Para el análisis de sentimientos se han entrenado dos modelos, el **LDA** y **NMF** proporcionadoa por la librería sklearn. Tras ver los resultados de topics, se decidió escoger el modelo LDA ya que ofrecía una mayor precisión a la hora de categorizar los tweets. 
Para poder entrenar este modelo, se utilizo un dataset de artículos del NYT, los cuales fueros preprocesados, eliminando caracteres indeseados, y posteriormente fueron tokenizados, lematizados, y pasados a una matriz Term Frequency. Se eliminaron stop words y se decidió crear 1000 features, para posteriormente asignarlas a 15 topics. 
Para optimizar nuestro modelo, se realizo un GridSearch sobre los parámetros de LDA, y conseguimos asignar una learning Rate adecuada.
Tras el entrenamiento, obtuvimos 15 topics, que en base a las palabras que mejor los describían (las 10 con mayor frecuencia), logramos clasificar los temas en un dataframe, que nos serviría posteriormente a la hora de asignar temas a nuevos documentos/tweets.

* ## Análisis de Sentimientos
El análisis de sentimientos en cambio no tuvo entrenamiento previo con dataset, sino que se utilizo la librería **Textblob**, que pasándole los nuevos tweets, los da un score [-1, 1] que clasificaría a los tweets en negativos si es menor que 0, en positivos si es mayor y en neutrales si es igual a 0. 


