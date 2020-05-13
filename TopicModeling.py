from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import time
import pickle
from tqdm import tqdm

# Importamos el dataset de entrenamiento de New York Times News y lo dividimos por artículo
with open('./files/nytimes_news_articles.txt', encoding="utf8") as txt:
    data = txt.read().split("\n\n")

# Hacemos una limpieza de los artículos
data = [x.strip().strip('\n').strip("'") for x in data]
document_cleaning = [x for x in data if x[:3]!="URL" and len(x)>0]  #eliminamos el texto que sea una URL o no tenga valores
document_cleaning = list(map(lambda x: re.sub('[\'“”’\-:—;\(\),\.!?\d]', '', x), document_cleaning))  #quitamos signos de puntuación
document_cleaning = list(map(lambda x: x.lower(), document_cleaning))  # pasamos el texto a minúsculas


# Primero tokenizamos los documentos en palabras
tokenized_documents = [word_tokenize(i) for i in document_cleaning]
with open("./files/tokenized_documents.pkl", 'wb') as f:
    pickle.dump(tokenized_documents, f)

# Lematizamos todas las palabras y eliminamos las stop words

lemmatized_documents = []
stop_words = stopwords.words()

for i in tqdm(range(len(tokenized_documents))):
    lemmatized_documents.append(list())
    for word in tokenized_documents[i]:
        if word not in stop_words:
            try:
                lemmatized_documents[i].append(WordNetLemmatizer.lemmatize(word))
            except:
                lemmatized_documents[i].append(word)
    time.sleep(0.009)

with open("./files/lemmatized_documents.pkl", 'wb') as f:
    pickle.dump(lemmatized_documents, f)


no_features = 1000

# tf-idf para NMF, ya que es un modelo matemático basado en algebra lineal.
# Se aplica un Preprocessor que una las palabras de cada documento antes de hacer la matriz tf-idf
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, preprocessor=' '.join,
                                   max_features=no_features, stop_words='english', lowercase=False)
tfidf = tfidf_vectorizer.fit_transform(lemmatized_documents)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

with open('files/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)
# # tf para LDA porque es un modelo probabilístico.
# # Se aplica el mismo preprocessor que en tf-idf, ya que tenemos los documentos tokenizados
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, preprocessor=' '.join, lowercase=False,
                                 max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(lemmatized_documents)
tf_feature_names = tf_vectorizer.get_feature_names()

with open('files/tf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tf_vectorizer, f)
with open('./files/feature_names.pkl', 'wb') as f:
    pickle.dump(tf_feature_names, f)

# # Entrenamos los modelo NMF y LDA con el tf-idf para NMF y tf para LDA
# # # Ejecutamos el modelo NMF
nmf = NMF(n_components=15, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
# #
# # # # Guardamos el modelo NMF
with open('./files/nmf.pkl', 'wb') as f:
    pickle.dump(nmf, f)
# # # Ejecutamos modelo LDA
lda = LatentDirichletAllocation(n_components=15, max_iter=5, learning_method='online', learning_offset=50.,
                                  random_state=0, learning_decay=0.5).fit(tf)
# # # # # Guardamos el modelo LDA
with open('./files/lda.pkl', 'wb') as f:
    pickle.dump(lda, f)

# Creamos una función para mostrar los topics que ha creado cada modelo
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))


no_top_words = 10

display_topics(nmf, tfidf_feature_names, no_top_words)
display_topics(lda, tf_feature_names, no_top_words)
