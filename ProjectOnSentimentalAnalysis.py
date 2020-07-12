import twitter
import re
import nltk
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
nltk.download('punkt')
#nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from string import punctuation 
from nltk.corpus import stopwords 

twitter_api = twitter.Api(consumer_key='**************************',
                        consumer_secret='*******************************',
                        access_token_key='********************************',
			access_token_secret='****************************')

#print(twitter_api.VerifyCredentials())

def buildTestSet(search_keyword):
    try:
        tweets_fetched = twitter_api.GetSearch(search_keyword, count = 150)
        
        print("Fetched " + str(len(tweets_fetched)) + " tweets for the term " + search_keyword)
        
        return [{"text":status.text, "label":None} for status in tweets_fetched]
    except:
        print("Unfortunately, something went wrong..")
        return None

search_term = "Amazon"
testDataSet = buildTestSet(search_term)

#stopwords = open(r'C:\\Users\\sys\\Anaconda3\\stopwords\\english")

class PreProcessTweets:
    def __init__(self):
        self._stopwords = set(stopwords.words('english') + list(punctuation) + ['AT_USER','URL'])
        self.emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)

    def processTweets(self, list_of_tweets):
        processedTweets=[]
        for tweet in list_of_tweets:
            processedTweets.append((self._processTweet(tweet["text"]),tweet["label"]))
        return processedTweets
    
    def _processTweet(self, tweet):
       
        tweet = self.emoji_pattern.sub(r'',tweet) # remove emoji
        tweet = tweet.lower() # convert text to lower-case
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet) # remove URLs
        tweet = re.sub('@[^\s]+', 'AT_USER', tweet) # remove usernames
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet) # remove the # in #hashtag
        #tweet = re.sub(self.emoji_pattern, r'\1', tweet)
        
        tweet = word_tokenize(tweet) # remove repeated characters (helloooooooo into hello)
        
        return [word for word in tweet if word not in self._stopwords]
    
training = pd.read_csv(r"C:\\Users\\sys\Anaconda3\\full-corpus.csv")
training = shuffle(training)


#training = training[~training['label'].str.contains('neutral')]
#training = training[~training['label'].str.contains('irrelevant')]

TrainingText = list(training["text"])
TrainingLabel = list(training["label"])
n = len(TrainingText)
trainingData = []
for i in range(n):
    trainingData.append({"text":TrainingText[i],"label":TrainingLabel[i]})

tweetProcessor = PreProcessTweets()

processedTraining = tweetProcessor.processTweets(trainingData)
processedTestData = tweetProcessor.processTweets(testDataSet)

processedTrainingData,test = train_test_split(processedTraining,test_size=0.1,random_state=None)
#print(processedTestData)
#processedTrainingData = processedTrainingData[160:180]
#print(processedTrainingData)

def buildVocabulary(preprocessedTrainingData):
    all_words = []
    
    for (words, sentiment) in preprocessedTrainingData:
        all_words.extend(words)

    wordlist = nltk.FreqDist(all_words)
    word_features = wordlist.keys()
    
    return word_features


word_features = buildVocabulary(processedTrainingData)
#print(word_features)
def extract_features(tweet):
    tweet_words=list(tweet)
    features={}
    for word in word_features:
        features['contains(%s)' % word]=(word in tweet_words)
    return features 

#features = extract_features(processedTrainingData)
print()
trainingFeatures=nltk.classify.apply_features(extract_features,processedTrainingData)
#print(trainingFeatures)

NBayesClassifier=nltk.NaiveBayesClassifier.train(trainingFeatures)
ActualValues = []
for (word,sentiment) in test:
    ActualValues.append(sentiment)
    
NBResultLabels = [NBayesClassifier.classify(extract_features(tweet[0])) for tweet in test]
print(accuracy_score(ActualValues,NBResultLabels))

if NBResultLabels.count('positive') > NBResultLabels.count('negative') :
    print("Overall Positive Sentiment")
    print("Positive Sentiment Percentage = " + str(100*NBResultLabels.count('positive')/len(NBResultLabels)) + "%")
elif NBResultLabels.count('positive') == NBResultLabels.count('negative') :
    print("Overall Neutral Sentiment")
else:
    print("Overall Negative Sentiment")
    print("Negative Sentiment Percentage = " + str(100*NBResultLabels.count('negative')/len(NBResultLabels)) + "%")

