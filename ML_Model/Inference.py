import os
import pickle
import pandas as pd
import re 
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer

# Reading the datasets
print('1. Reading ml models and inputs.... ')
current_directory = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_directory, "ML_Model_Package.pkl")
with open(model_path, 'rb') as f:
    model= pickle.load(f)

def infer(inputs):
    X_input= inputs
    cols= ['Age', 
        'Feeling sad or Tearful', 
        'Irritable towards baby & partner',
        'Trouble sleeping at night',
        'Problems concentrating or making decision',
        'Overeating or loss of appetite', 
        'Feeling anxious', 
        'Feeling of guilt',
        'Problems of bonding with baby', 
        'Suicide attempt']

    # 2. Stemming (converting words to its Root words)+ remove Stop words
    print('2. Performing Stemming.... ')
    stopwords= ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    port_stem= PorterStemmer()
    def stemming(content):
        stemmed_content= re.sub('[^a-zA-Z]', ' ', content)
        stemmed_content= stemmed_content.lower()
        stemmed_content= stemmed_content.split()
        stemmed_content= [port_stem.stem(word) for word in stemmed_content if not word in stopwords]
        stemmed_content= ' '.join(stemmed_content)
        return stemmed_content
    for col in cols:
        if (col!='Age'):
            X_input[col]= X_input[col].apply(stemming)

    # 3. Vectorization (convert text to numerical equivalents)
    print('4. Performing Vectorization.... ')
    vectorizer= TfidfVectorizer()
    def vectorize(text): # return score between 0 to 5
        text= text.split()
        temp= text
        temp= vectorizer.fit_transform(temp).toarray()
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
            X_input[col]= X_input[col].apply(vectorize)
            
    # 4. Inference
    print('4. Inference....')
    y_predicted= model.predict(X_input)
    return y_predicted

if __name__=="__main__":
    input= pd.DataFrame({'Age':[42.5], 
        'Feeling sad or Tearful':['Yes'], 
        'Irritable towards baby & partner':['No'],
        'Trouble sleeping at night':['No'],
        'Problems concentrating or making decision':['Yes'],
        'Overeating or loss of appetite':['Yes'], 
        'Feeling anxious':['No'], 
        'Feeling of guilt':['Yes'],
        'Problems of bonding with baby':['Yes'], 
        'Suicide attempt':['No']})
    prediction2= infer(input)
    print(prediction2)
