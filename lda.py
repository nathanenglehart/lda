#!/usr/bin/env python3

import pandas as pd
import numpy as np
import string

np.random.seed(123)
verbose = True

punc = string.punctuation

########
# AI generated text content
########

documents = [
    "apple banana apple fruit orange banana fruit apple orange",
    "dog cat mouse pet dog cat dog mouse pet dog cat",
    "car engine wheel car road vehicle engine car road wheel vehicle",
    "apple fruit orange banana apple fruit orange banana apple fruit",
    "dog pet cat mouse dog pet cat mouse dog pet cat mouse dog",
    "engine car road wheel engine car road wheel engine car road wheel",
    "fruit banana apple orange fruit banana apple orange fruit banana apple orange",
    "pet mouse cat dog pet mouse cat dog pet mouse cat dog pet mouse",
    "wheel vehicle car engine wheel vehicle car engine wheel vehicle car engine",
    "banana apple orange fruit banana apple orange fruit banana apple orange fruit banana",
    "cat dog mouse pet cat dog mouse pet cat dog mouse pet cat dog mouse pet",
    "car road engine wheel car road engine wheel car road engine wheel car road engine",
    "orange banana apple fruit orange banana apple fruit orange banana apple fruit orange banana",
    "mouse cat dog pet mouse cat dog pet mouse cat dog pet mouse cat dog pet mouse cat",
    "vehicle car engine wheel vehicle car engine wheel vehicle car engine wheel vehicle car engine",
    "apple orange fruit banana apple orange fruit banana apple orange fruit banana apple orange fruit banana",
    "pet dog cat mouse pet dog cat mouse pet dog cat mouse pet dog cat mouse pet dog",
    "wheel engine car road wheel engine car road wheel engine car road wheel engine car road"
]

K = 3
df = pd.DataFrame({'text': documents})
df['text'] = df['text'].str.replace('[{}]'.format(''.join(p for p in punc)), '', regex=True).str.lower()

########
# Latent Dirichlet Allocation 
########

class lda():
    
    def __init__(self, df, K, alpha = 0.1, beta = 0.01, epoch = 1000): 

        self.df = df.copy()
        self.df['text'] = self.df['text'].str.split()
        self.df['word-count'] = self.df['text'].apply(lambda x : len(x)) # Computes n_{i,•} (i.e. # words in each doc)

        self.vocabulary = set([word for doc in self.df['text'] for word in doc])
        self.V = len(self.vocabulary) 

        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.epoch = epoch

    def randomly_assign_topics(self, text, K):

        """ Randomly assigned each word in each doc to one of 1,...,K topics. """

        topics = np.arange(1,K+1)
        num_topics = len(topics)
        assigned_topics = np.random.choice(topics, size=len(text))
        return assigned_topics

    def get_curr_word_topic_count(self, df):

        """ Computes n_{k,w_{i,j}}, i.e. the number of times word w_{i,j} is assigned to topic k across all docs. """

        df_exploded = df.explode(['text', 'word-topic'])
        df_counts = df_exploded.groupby(['text', 'word-topic']).size().reset_index(name='count')
        df_pivot = df_counts.pivot(index='text', columns='word-topic', values='count').fillna(0)
        return df_pivot.to_dict(orient='index')

    def get_num_topic_words_in_doc(self, df): 

        """ Computes n_{i,k}, i.e. the number of words assigned to topic k in doc i. """

        df_exploded = df.explode('word-topic')
        counts = df_exploded.groupby([df_exploded.index, 'word-topic']).size().reset_index(name='count')
        df_pivot = counts.pivot(index='level_0', columns='word-topic', values='count').fillna(0).astype(int)
        df['num-topic-words'] = df_pivot.apply(lambda row: row.to_dict(), axis=1)    
        return df

    def get_num_words_per_topic(self, word_topic_count, K):
        
        """ Computes n_{k,•}, i.e. the total number of words assigned to each topic across all documents. """

        num_words_per_topic = np.zeros(K) 

        for words, word_topic_counts in word_topic_count.items():
            for k in range(1,K+1):
                num_words_per_topic[k-1] += word_topic_counts[k]

        return num_words_per_topic

    def compute_probabilities(self, words, word_topic_count, num_words_per_topic, num_topic_words, word_count, word_to_index):

        """ Computes the posterior probabilities that a word belongs to a topic given the prior topic and the word. 

            Args:
                
                    words::[List]
                        List of strings where each entry is a word in the given document in sequential order.

                    word_topic_count::[Dict]
                        Dictionary with word keys and dictionary values. Dictionary values have topic keys and count values.

                    num_words_per_topic::[Numpy Array]
                        Numpy array with integer values corresponding to the word counts for each topic.

                    word_count::[Integer]
                        Number of words in the given document (which words corresponds to).

        """

        probabilities = []

        for word in words:

            word_index = word_to_index.get(word, -1) # rewrite this

            if word_index == -1:
                prob = np.zeros(K)
            else:
                prob = np.zeros(K)

                for k in range(K): 
                    prob[k] = ((num_topic_words[k+1] + self.alpha) / (word_count + self.K * self.alpha) ) * \
                              ((word_topic_count[word][k+1] + self.beta) / (num_words_per_topic[k] + self.V * self.beta))

                prob /= prob.sum()

            probabilities.append(prob.tolist())

        return probabilities

    def gibbs(self):

        """ Uses the Gibbs Sampling MCMC technique to approximate the posterior topic probabilities for each word. """

        # source: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC387300/
        # epoch times:
        #   * for each word in each document, compute P(z_{i,j} = k | z_{i,j}, w)
        #   * then sample from the computed posterior and update ...

        self.df['word-topic'] = self.df['text'].apply(lambda doc : self.randomly_assign_topics(doc, K))
        word_topic_count = self.get_curr_word_topic_count(self.df)
        self.df = self.get_num_topic_words_in_doc(self.df)

        num_words_per_topic = self.get_num_words_per_topic(word_topic_count, K)
        word_to_index = {word: idx for idx, word in enumerate(word_topic_count.keys())} # could remove certain words

        for i in range(1, self.epoch):

            if(verbose):
                print("Running iteration " + str(i) + ".")

            # compute posterior

            self.df['prob-word-topic'] = self.df.apply(lambda row: self.compute_probabilities(row['text'],
                                                                                              word_topic_count,
                                                                                              num_words_per_topic,
                                                                                              row['num-topic-words'],
                                                                                              row['word-count'],
                                                                                              word_to_index), 
                                                                                              axis=1)

            # sample from posterior

            self.df['word-topic'] = self.df['prob-word-topic'].apply(lambda ps: [np.random.choice(K, p=p) + 1 for p in ps])

            # update 

            self.df = self.get_num_topic_words_in_doc(self.df)
            word_topic_count = self.get_curr_word_topic_count(self.df)
            num_words_per_topic = self.get_num_words_per_topic(word_topic_count, K)

        self.df['word-topic'] = self.df['word-topic'].apply(np.array)

        return self
   
    def get_document_topic_prob(self, doc):
        return (list(doc['num-topic-words'].values()) + np.full(self.K,self.alpha)) / (doc['word-count'] + self.K * self.alpha)

    def get_document_topic_probs(self):
        
        """ Computes document-topic probabilities (i.e. what proportion of the document is about which topic. """

        self.df['document-topic'] = self.df.apply(self.get_document_topic_prob, axis = 1).apply(np.array)

        return self

########
# Running the model
########

model = lda(df = df, K = 3, epoch = 200)
model = model.gibbs()
model = model.get_document_topic_probs()

df_lda = model.df[['text','document-topic','word-topic']]
df_lda.to_csv("./results.csv")

