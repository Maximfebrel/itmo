import pandas as pd
import re
import spacy
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

nlp = spacy.load("en_core_web_sm")


class Loader:
    def __init__(self, num_rows=None):
        self.data = pd.read_csv(r"C:\Users\makso\Desktop\ФООСИИ\transformer\russian_comments_from_2ch_pikabu.csv")[0: 6000]
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.num_rows = num_rows

        self.make_clear_text()
        self.vectorize()

    @staticmethod
    def lemmatize(text):
        global nlp
        doc = nlp(text)
        return " ".join([token.lemma_ for token in doc])

    @staticmethod
    def clear(text):
        return " ".join(re.sub(r'[^a-zA-Z]', ' ', text.lower()).split())

    def make_clear_text(self):
        self.data['clear_text'] = self.data['translated'].apply(self.clear)
        self.data['lemmatized_text'] = self.data['clear_text'].apply(self.lemmatize)

    def vectorize(self):
        train, test = train_test_split(self.data, test_size=0.2, random_state=12345)

        corpus_train = train['lemmatized_text']
        corpus_test = test['lemmatized_text']

        nltk.download('stopwords')
        stopwords = nltk.corpus.stopwords.words('english')

        count_tf_idf = TfidfVectorizer(stop_words=stopwords)

        self.X_train = count_tf_idf.fit_transform(corpus_train)
        self.X_test = count_tf_idf.transform(corpus_test)

        self.y_train = train['toxic']
        self.y_test = test['toxic']

