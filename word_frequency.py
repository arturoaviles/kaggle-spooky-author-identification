import nltk
import pandas as pd

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

for _, row in texts_groupby_author.iterrows():
    author = row["author"]
    text = row["text"]
    tokens = nltk.tokenize.word_tokenize(text)
    frequency = nltk.FreqDist(tokens)
    word_frequencies_by_author[author] = frequency

# Test
sentence = "Still, as I urged our leaving Ireland with such inquietude and impatience, my father thought it best to yield."
sentence = sentence.lower()
sentence_tokens = nltk.tokenize.word_tokenize(sentence)

for author in word_frequencies_by_author.keys():
    total = 0
    for word in sentence_tokens:
        total += word_frequencies_by_author[author].freq(word) 
    print(total, author)

# End test

# Create a new dataframe to save the data
dataframe_with_frequencies = pd.DataFrame(columns=('id', 'EAP', 'HPL', 'MWS'))
dataframe_with_frequencies.head()


# Open the test file and iterate on it
test = pd.read_csv("data/test.csv")
for iter_num, row in test.iterrows():
    sentence = row["text"] # Get sentence
    sentence = sentence.lower() # Str to lower
    sentence_tokens = nltk.tokenize.word_tokenize(sentence) # Tokenize test words
    
    # Get the author and probability of authorship attribution
    row_results = [row["id"]]
    for author in word_frequencies_by_author.keys():
        total = 0
        for word in sentence_tokens:
            total += word_frequencies_by_author[author].freq(word) 
        #print(total, author)
        total = total / len(sentence_tokens)
        row_results.append(total)
        
    # Add a new row to the dataframe
    dataframe_with_frequencies.loc[iter_num+1] = row_results

# Save the dataframe to a CSV file
dataframe_with_frequencies.to_csv("word_frequency.csv", index=False)