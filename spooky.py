import nltk
import pandas as pd

nltk.download('punkt')

""" There are many features to determine if a text is from an author, for example:

- Word Frecuency
- Vocabulary Richness
- Verb C
- N-Grams
- etc.

In this case, I will use Word Frecuency
"""

# 1st, I will check the available data. This data contains the texts with their respective Author. 
texts_with_author = pd.read_csv("data/train.csv")
texts_with_author.head()

# The ids are not important, we need to group and join all the texts by author
# The stop words inside this texts will help beacuse we are not trying to understand the intetion of the author.
# Every person uses this stop words with more or less frecuency than others.
texts_groupby_author = texts_with_author.groupby("author")["text"].apply(' '.join).reset_index()
texts_groupby_author

# We transform every text to lowercase to eliminate the tokenization of same words with capitalized letters.
texts_groupby_author["text"] = texts_groupby_author.text.str.lower()
texts_groupby_author

# We can use a dict to save the word frequencies for each author
word_frequencies_by_author = {}

# Iterate through the data
for _, row in texts_groupby_author.iterrows():
    author = row["author"] # Get row author
    text = row["text"] # Get row text
    tokens = nltk.tokenize.word_tokenize(text) # Create list of words, also having periods
    frequency = nltk.FreqDist(tokens) # Get the frequency of every token in the text 
    word_frequencies_by_author[author] = frequency # Save the frequency for each author


# Simple test to check if it works
sentence = "Still, as I urged our leaving Ireland with such inquietude and impatience, my father thought it best to yield."
sentence = sentence.lower()
sentence_tokens = nltk.tokenize.word_tokenize(sentence)

for author in word_frequencies_by_author.keys():
    total = 0
    for word in sentence_tokens:
        total += word_frequencies_by_author[author].freq(word) 
    print(total, author)
# End of test


# Create a new dataframe to save the data
dataframe_with_frequencies = pd.DataFrame(columns=('author', 'frequency', 'sentence'))
dataframe_with_frequencies.head()


# Open the test file and iterate on it
test = pd.read_csv("data/test.csv")
for iter_num, row in test.iterrows():
    sentence = row["text"] # Get sentence
    sentence = sentence.lower() # Str to lower
    sentence_tokens = nltk.tokenize.word_tokenize(sentence) # Tokenize test words
    
    # Get the author that has more probability to be matched with the text
    best_frequency_author = ""
    best_frequency_by_author = 0
    for author in word_frequencies_by_author.keys():
        total = 0 # counter to sum all the word probabilities of the sentence by author
        for word in sentence_tokens: # iterate through the sentence 
            total += word_frequencies_by_author[author].freq(word) 
        #print(total, author)
        if(total > best_frequency_by_author):
            best_frequency_author = author
            best_frequency_by_author = total
        
    # Add a new row to the dataframe
    dataframe_with_frequencies.loc[iter_num+1] = [best_frequency_author, best_frequency_by_author, sentence]


# Save the dataframe to a CSV file
dataframe_with_frequencies.to_csv("results2.csv")


