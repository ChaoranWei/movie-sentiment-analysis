from sklearn.ensemble import RandomForestClassifier 
from sklearn.feature_extraction.text import CountVectorizer
import re
import pandas as pd   
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup  
import re
import numpy as np

'''This module imitates the tutorial for "bag of popcoren" kaggle competition.
The formation of word vectors are changed a little bit. Instead of using random
forest, I implemented several different algorithms as well.
'''

train = pd.read_csv("labeledTrainData.tsv", header=0, \
                    delimiter="\t", quoting=3)
#header = 0 means first line is title, quoting means ignore double quotes

def review_to_words( raw_review ):
   
    
    review_text = BeautifulSoup(raw_review).get_text()  #remove hmtl tage

    letters_only = re.sub("[^a-zA-Z]", " ", review_text) #remove symbols except letters
    words = letters_only.lower().split()                             
    stops = set(stopwords.words("english"))                  

    meaningful_words = [w for w in words if not w in stops]   #remove stopwords
    return( " ".join( meaningful_words ))   #get one string per review

num = train["review"].size

clean_version = []

for i in xrange( 0, num ):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    clean_version.append( review_to_words( train["review"][i] ) )
    
#print clean_version

print "Creating the bag of words...\n"

vectorizer = CountVectorizer(analyzer = 'word',
                             tokenizer = None,
                             preprocessor = None,
                             stop_words = None,
                             max_features = 6000) #it is able to do data preprocessing for us

features = vectorizer.fit_transform(clean_version) #this gets a 5000 times 25000 matrix
features = features.toarray()

vocab = vectorizer.get_feature_names()
print vocab

#np.save('pickle/word_list.npy', features)

# Sum up the counts of each vocabulary word
dist = np.sum(features, axis=0)

# For each, print the vocabulary word and the number of times it 
# appears in the training set
for tag, count in zip(vocab, dist):
    print count, tag


# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 500) 
    
test = pd.read_csv("testData.tsv", header=0, delimiter="\t", \
                   quoting=3 )
train = pd.read_csv("labeledTrainData.tsv", header=0, \
                    delimiter="\t", quoting=3)
print 'loaded data'
forest = forest.fit( features, train["sentiment"] )
print 'fitted to the forest'
# Verify that there are 25,000 rows and 2 columns
print test.shape

# Create an empty list and append the clean reviews one by one
num_reviews = len(test["review"])
clean_test_reviews = [] 

print "Cleaning and parsing the test set movie reviews...\n"
for i in xrange(0,num_reviews):
    if( (i+1) % 1000 == 0 ):
        print "Review %d of %d\n" % (i+1, num_reviews)
    clean_review = review_to_words( test["review"][i] )
    clean_test_reviews.append( clean_review )

test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

result = forest.predict(test_data_features)

output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

# Use pandas to write the comma-separated output file
output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )    