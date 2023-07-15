import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()

#Preprocesing function
stop_words_list= ["a", "the" , "to" , "be" ,"is" , "an", "hi", "hello", "good morning","goodmorning",'goodevening',
                  'good evening', "hey", "wow", "buyer", "seller", "trader", 'good', 'noon','morning','evening', 'here',
                  "here recordingenjoy", "all", "everyone", "any", "help", "pls", "please", "certainely", "fee","like" ,"comment", "comment",
                 "friends","friend", "wi", "only", "verify", "asap", "some", "me", "fuck", "lol", "anyone", "cmnt", "omg", "wow","wtf", "oo",
                 "everyone", "page", "huh", "any", "new", "have", "my", "u", "attend", "look", "better", "interest"]

def lemmatize_word(text): 
    from nltk.tokenize import word_tokenize 
    from nltk.stem import WordNetLemmatizer
    lemmatizer=WordNetLemmatizer()
    word_tokens = word_tokenize(text) 
    lemmas = [lemmatizer.lemmatize(word,pos='v') for word in word_tokens] 
    return lemmas

def clean_text(text,stop_words=None):
    text = text.lower().strip() # Convert text to lower case and strip leading/trailing white spaces
    text = text.lower().strip() # Convert text to lower case and strip leading/trailing white spaces
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'http\S+', '', text) # Remove websites
    text = re.sub(r'[^\w\s$]', '', text) # Remove punctuation except $
    text= re.sub(r"cant",'can not',text)
    text= re.sub(r"didnt","did not",text)
    text= re.sub(r"dont","do not",text)
    text= re.sub(r"wasnt","was not",text)
    text= re.sub(r"wont","will not",text)    
    text= re.sub('\S+@\S+','',text) #removing emailadderess
    text= re.sub(r'https?:\/\/.*[\r\n]*', ' ', text) # removing all urls
    text= re.sub(r"\[.*?\]", " ",text)# removing data in square brackets
    text= re.sub("[^a-z\s$]+"," ",text)
    text= re.sub("thank you|thank you soo much|thank you so much|thanks a lot|thank you very much","thanks",text)
    text= re.sub(r"(\.|,|!)([a-zA-Z0-9]{1})",r"\1 \2",text)
    text= re.sub(' +',' ',text)  # removing extra spaces
    text= text.replace(",", "")
    text= text.replace(".", "")
    text= text.replace("?", "")
    text= text.replace("\n"," ")
    text= text.replace("'pm"," contact")# expansion contact
    text= text.replace("'call me"," contact")# expansion contact
    text= text.replace("'m"," am")# expansion am
    text= text.replace("'ll"," will")# expansion will
    text= text.replace("'d"," would")# expansion would
    text= text.replace("n't"," not")# expansion not
    text= text.replace("'ve"," have")# expansion have
    text= text.replace("'s"," is")# expansion is
    text= text.replace("'re"," are")# expansion are
    text= text.replace("'prize"," price")# expansion am
    text= text.replace("ain't","are not")
    text= text.replace("aint","are not")
    text= text.replace("cause","because")
    text= text.replace("couldn't","could not")
    text= text.replace("couldnt","could not")
    text= text.replace("didn't","did not")
    text= text.replace("didnt","did not")
    text= text.replace("doesn't","does not")
    text= text.replace("doesnt","does not")
    text= text.replace("don't","do not")
    text= text.replace("dont","do not")
    text= text.replace("everything's","everything is")
    text= text.replace("everythings","everything is")
    text= text.replace("everyones","everyone is")
    text= text.replace("everyone's","everyone is")
    text= text.replace("haven't","have not")
    text= text.replace("havent","have not")
    text= text.replace("hasn't","has not")
    text= text.replace("hasnt","has not")
    text= text.replace("hadn't","had not")
    text= text.replace("hadnt","had not")
    text= text.replace("he's","he is")
    text= text.replace("hes","he is")
    text= text.replace("i'll","i will")
    text= text.replace("ill","i will")
    text= text.replace("i'm","i am")
    text= text.replace("im","i am")
    text= text.replace("it's","it is")
    text= text.replace("its","it is")
    text= text.replace("i've","i have")
    text= text.replace("ive","i have")
    text= text.replace("let's","let us")
    text= text.replace("lets","let us")
    text= text.replace("she's","she is")
    text= text.replace("shes","she is")
    text= text.replace("shouldn't","should not")
    text= text.replace("shouldnt","should not")
    text= text.replace("that's","that is")
    text= text.replace("thats","that is")
    text= text.replace("there's","there is")
    text= text.replace("theres","there is")
    text= text.replace("they're","they are")
    text= text.replace("theyre","they are")
    text= text.replace("they've","they have")
    text= text.replace("theyve","they have")
    text= text.replace("wasn't","was not")
    text= text.replace("wasnt","was not")
    text= text.replace("we'd","we would")
    text= text.replace("wed","we would")
    text= text.replace("we'll","we will")
    text= text.replace("well","we will")
    text= text.replace("we're","we are")
    text= text.replace("were","we are")
    text= text.replace("weren't", "were not")
    text= text.replace("werent", "were not")
    text= text.replace("what's", "what is")
    text= text.replace("whats", "what is")
    text= text.replace("who's", "who is")
    text= text.replace("whos", "who is")
    text= text.replace("won't", "will not")
    text= text.replace("wont", "will not")
    text= text.replace("wouldn't", "would not")
    text= text.replace("wouldnt", "would not")
    text= text.replace("you'll", "you will")
    text= text.replace("youll", "you will")
    text= text.replace("you're", "you are")
    text= text.replace("youre", "you are")
    text= text.replace("you've", "you have")
    text= text.replace("youve", "you have")
    text= text.replace("kiss", "case")
    text= text.replace("gonna", "going to")
    text= text.replace("yeah", "yes")
    text= text.replace("may", "my")
    text= text.replace("ise", "is")
    text= text.strip()
    text= text.replace('uhhuh', '')
    text= lemmatize_word(text)
    if stop_words is None:
        stop_words= stop_words_list
    text = [word for word in text if word not in stop_words]
    return text
def remove_emoji(text):
    # Regular expression to match emoji
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)