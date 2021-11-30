
import pickle 
import tflearn
import nltk
import numpy
import json

from os import system, name
from tensorflow.python.framework import ops
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

class ChatBot:

    def __init__(self):
        ''' Initializes the model and model information to be used during classification. '''

        with open('data.json') as file:
            data = json.load(file)

        self.responses = {}

        for intent in data['intents']:
            self.responses[intent["label"]] = intent["response"]

        self.model_information = self.load("model_information")
        
        self.X_train = numpy.array(self.model_information["X_train"])
        self.y_train = numpy.array(self.model_information["y_train"])

        ops.reset_default_graph()

        net = tflearn.layers.core.input_data(shape=[None, len(self.X_train[0])])
        net = tflearn.layers.core.fully_connected(net, 8)
        net = tflearn.layers.core.fully_connected(net, 8)
        net = tflearn.layers.core.fully_connected(net, len(self.y_train[0]), activation="softmax")
        net = tflearn.layers.estimator.regression(net)

        self.model = tflearn.DNN(net)
        self.model.load('./model.tflearn')

    def clear(self):
        ''' Function used to clear the terminal. '''
        
        if name == 'nt':
            _ = system('cls')
        else:
            _ = system('clear')
            
    def load(self, sFilename):
        ''' Given a file name, load and return the object stored in the file. '''

        f = open(sFilename, "rb")
        u = pickle.Unpickler(f)
        dObj = u.load()
        f.close()
        return dObj

    def preprocess(self, text):
        ''' Accepts text as a parameter and performs the following preprocessing: 1. Tokenization
        2. Lemmatization 3. Convert words to lower case. Returns preprocessed tokens as output.
        Stop words and punctuations provide valuable context and hence are not preprocessed. '''

        tokens = []
        words = nltk.word_tokenize(text)

        for word in words:
            tokens.append(lemmatizer.lemmatize(word.lower()))

        return tokens

    def bag_of_words(self, text, tokens):
        ''' Accepts a string and returns the bag of words. '''

        bag = [0 for _ in range(len(tokens))]
        words = self.preprocess(text)

        for word in words:

            for i, token in enumerate(tokens):
                if word == token:
                    bag[i] = 1

        return numpy.array(bag)

    def chat(self):
        ''' This function classifies the user input and generates appropriate response. '''

        self.clear()

        user = input("Please enter your name to begin: ")
        print("Start talking with the bot (type exit to stop)")

        tokens = self.model_information["tokens"]
        labels = self.model_information["labels"]

        while True:

            user_input = input(user + ": ")

            if(user_input.lower() == "exit"):
                break

            results = self.model.predict([self.bag_of_words(user_input, tokens)])
            index = numpy.argmax(results)
            label = labels[index]

            # print("Classified as: " + labels[index])

            print("Foodie: " + self.responses[label])

            
cb = ChatBot()
cb.chat()          


            
        





