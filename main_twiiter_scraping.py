# MS548 Sentiment Anylysis Assignment
# Kenneth Carr

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np
from tkinter import *

root = Tk()
root.title("Tweet Anylyzer - Carr")

def myClickSentiment():
    tweet = user_tweet_input.get() # gets input from textbox from user
    txt_log_file = open('event_log_file.txt','a')
    txt_log_file.write("user_input: " + tweet + "\n")
    txt_log_file.close()
    tweet_words = [] # list to store processed tweet
    for word in tweet.split(' '): # remove content that is not part of sentence
        if word.startswith('@') and len(word) > 1:
            word = " "
        elif word.startswith('http'):
            word = " "
        elif word.startswith('#'):
            word=" "
        tweet_words.append(word)

    processed_tweet = " ".join(tweet_words) # combines list back into a sentance string

    # load model and tokenizer
    roberta_dataset = "cardiffnlp/twitter-roberta-base-sentiment"

    training_model = AutoModelForSequenceClassification.from_pretrained(roberta_dataset)
    tokenizer = AutoTokenizer.from_pretrained(roberta_dataset)

    labels = ['Negative Tweet', 'Neutral Tweet', 'Positive Tweet'] # Label outputs based on position in list

    # sentiment analysis
    encoded_tweet = tokenizer(processed_tweet, return_tensors='pt')
    output = training_model(encoded_tweet['input_ids'], encoded_tweet['attention_mask'])

    sentiment_scores = output[0][0].detach().numpy()
    sentiment_scores = softmax(sentiment_scores)
    sentiment_scores = np.round(sentiment_scores, 4)

    confidence_level = 0.7 # looking for dominant characteristic
    for index in range(len(sentiment_scores)): 
        # 0 = negative, 1 = neutral, 2 = positive
        l_value = labels[index]
        s_value = sentiment_scores[index]
        #print(l_value,s_value, index)
        if(s_value > confidence_level):
            results_output = str(l_value) + " @ " + str(np.round((s_value * 100),2)) + "% Confidence Rating"
            print(results_output)
            tweet_assesment_label.config(text=results_output)
            txt_log_file = open('event_log_file.txt','a')
            txt_log_file.write("results: " + results_output + "\n")
            txt_log_file.close()

application_header_label = Label(root, text="Twitter Sentiment Anylysis", font=("Arial", 24))
application_header_label.pack(padx=10, pady=10)

user_tweet_input = Entry(root, width=50)
user_tweet_input.pack(padx=10, pady=10)
user_tweet_input.insert(0, "Enter Tweet Here")

tweet_assesment_label = Label(root, text="Enter Tweet To Anylyze")
tweet_assesment_label.pack(padx=10, pady=10)

anylyzeTweetButton = Button(root, text="Anylyze Tweet Sentiment", command=myClickSentiment)
anylyzeTweetButton.pack(padx=10, pady=10)

root.mainloop()