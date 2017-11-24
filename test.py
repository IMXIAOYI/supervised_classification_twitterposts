import sklearn
import pandas as pd

# load train dataset and text dataset
train_data = pd.read_csv('train.csv',sep='\t')

# organize data
label_names= ['feel','fact']
labels= train_data['label']

# extract features from text files
##tokenizing text
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
train_counts= count_vect.fit_transform(train_data['text'])

##from occurrences to frequencies
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
train_tfidf = tfidf_transformer.fit_transform(train_counts)
train_tfidf.shape

#text data prep before predict
test_data= pd.read_csv('test.csv',sep='\t')
print(len(test_data))

from sklearn.linear_model import SGDClassifier
test_counts=count_vect.transform(test_data)
test_tfidf = tfidf_transformer.transform(test_counts)

#train a classifier svm
##build a pipeline 
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-3, random_state=42,
                                           max_iter=5, tol=None))])
text_clf.fit(train_data['text'],train_data['label'])

#classify result
predicted = text_clf.predict(test_data['text'])

#output
output = pd.DataFrame({"Id": range(0,4075), "Category": predicted})
output = output[['Id', 'Category']]
output.to_csv("output.csv", sep=',', index=False)