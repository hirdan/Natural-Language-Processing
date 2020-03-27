import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize('better','a'))

# first arguement is word
# second arguement is like whether it is adjective/noun/verb
# default arguement is noun
# lemmatizing is basically finding the synonym of the particular word.
