import os
import pickle
import pandas as pd
import re 
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier

# Reading the datasets
print('1. Reading Datasets.... ')
current_directory = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_directory, "post_natal_data_with_labels.csv")
df = pd.read_csv(file_path)
X= df.drop(columns=['Depressed'])
Y= df['Depressed']

# 2. Data Cleaning 
print('2. Cleaning Data.... ')
# convert age group from range to a flot value
def convert(text):
    a, b= text.split('-')
    a= int(a)
    b= int(b)
    return (a+b)/2
X['Age']= X['Age'].apply(convert)

# 3. Stemming (converting words to its Root words)+ remove Stop words
print('3. Performing Stemming.... ')
stopwords= ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
port_stem= PorterStemmer()
def stemming(content):
    stemmed_content= re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content= stemmed_content.lower()
    stemmed_content= stemmed_content.split()
    stemmed_content= [port_stem.stem(word) for word in stemmed_content if not word in stopwords]
    stemmed_content= ' '.join(stemmed_content)
    return stemmed_content
cols= X.columns
for col in cols:
    if (col!='Age'):
        X[col]= X[col].apply(stemming)


# 4. Vectorization (convert text to numerical equivalents)
print('4. Performing Vectorization.... ')
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify= Y, random_state=0)
def vectorize(text): # return score between 0 to 5
    text= text.split()
    yes_equil= ['ye', 'yeah', "affirmative", "aye", "indeed", "certainly", "absolutely", "sure", "yeah", "yep", "yah", "right", "positive", "agreed", "true", "correct", "affirm", "very well", "bet", "ofcourse"]
    often_equil= ['habitually', 'sometimes', 'generally', 'constantly', 'repeatedly', 'regularly', 'routinely', 'consistently', 'often', 'commonly', 'usually', 'oftentimes', 'frequently', 'typically', 'customarily']
    two_equil= ['two']
    one_equil= ['one']
    no_equil= ['no', 'not', "negative", "nay", "never", "nope", "nah", "deny", "refuse", "decline", "reject", "disagree", "veto", "negative", "non", 'ok', 'okay']

    for word in text:
        if word in yes_equil:
            return 5
        elif word in often_equil:
            return 4
        elif word in two_equil:
            return 2
        elif word in one_equil:
            return 1
        elif word in no_equil:
            return 0
    return 3
for col in cols:
    if (col!='Age'):
        X_train[col]= X_train[col].apply(vectorize)
        X_test[col]= X_test[col].apply(vectorize)

# 5. Train ML Model
print('5. Training ML Model.... ')
model= RandomForestClassifier()
model.fit(X_train, y_train)
y_predicted= model.predict(X_test)
print(y_predicted)

print('MSE: %.2f' % mean_squared_error(y_test, y_predicted))
print('r2: %.2f' % r2_score(y_test, y_predicted))

# from Inference import infer
# predictions= infer(xtest)
# print(predictions)
# 6. Save the Model
print('6. Saving ML Model.... ')
model_path = os.path.join(current_directory, f"ML_Model_Package.pkl")
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
