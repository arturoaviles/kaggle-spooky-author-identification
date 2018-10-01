import nltk
import pandas as pd


# 1st, I will check the available data. This data contains the texts with their respective Author. 
texts_with_author = pd.read_csv("data/train.csv")
texts_with_author.head()


# The ids are not important, we need to group and join all the texts by author
# The stop words inside this texts will help because we are not trying to understand the intention of the author.
# Every person uses this stop words with more or less frequency than others.
texts_groupby_author = texts_with_author.groupby("author")["text"].apply(' '.join).reset_index()
texts_groupby_author

# We transform every text to lowercase to eliminate the tokenization of same words with capitalized letters.
texts_groupby_author["text"] = texts_groupby_author.text.str.lower()
texts_groupby_author


# We can use a dict to save the bigrams frequencies for each author
bigrams_frequencies_by_author = {}

for _, row in texts_groupby_author.iterrows():
    author = row["author"]
    text = row["text"]
    tokens = nltk.tokenize.word_tokenize(text)
    bigrams = nltk.bigrams(tokens) # create bigrams of the text
    frequency_distribution = nltk.FreqDist(bigrams) # frequency distribution of all the bigrams in the text
    bigrams_frequencies_by_author[author] = frequency_distribution


# Print frequency distribution of author
print("EAP")
print(bigrams_frequencies_by_author["EAP"])

print("\nHPL")
print(bigrams_frequencies_by_author["HPL"])

print("\nMWS")
print(bigrams_frequencies_by_author["MWS"])


# Print all the bigrams and frequency by author
for author in bigrams_frequencies_by_author:
    print("\n\nAuthor: ", author)
    for bigram, frequency in frequency_distribution.items():
        print(bigram, frequency)


# Test 1
sentence = "Finding nothing else, not even gold, the Superintendent abandoned his attempts; but a perplexed look occasionally steals over his countenance as he sits thinking at his desk."
sentence = sentence.lower()
sentence_tokens = nltk.tokenize.word_tokenize(sentence)
sentence_bigrams = nltk.bigrams(sentence_tokens) # generator
sentence_bigrams_list = list(sentence_bigrams)

for author in bigrams_frequencies_by_author:
    total = 0
    for each_bigram in sentence_bigrams_list:
        total += bigrams_frequencies_by_author[author].freq(each_bigram) 
    print(total, author)


# Create a new dataframe to save the data
dataframe_with_frequencies = pd.DataFrame(columns=('id', 'EAP', 'HPL', 'MWS'))
dataframe_with_frequencies.head()


# Open the test file and iterate on it
test = pd.read_csv("data/test.csv")
for iter_num, row in test.iterrows():
    sentence = row["text"] # Get sentence
    sentence = sentence.lower() # Str to lower
    sentence_tokens = nltk.tokenize.word_tokenize(sentence) # Tokenize test words
    sentence_bigrams = nltk.bigrams(sentence_tokens) # generator
    sentence_bigrams_list = list(sentence_bigrams) # generator to list
    
    row_results = [row["id"]]
    for author in bigrams_frequencies_by_author: # iterate bigram freq by author
        total = 0 # total var in which freqs will be added
        for each_bigram in sentence_bigrams_list: # iterate each bigram
            total += bigrams_frequencies_by_author[author].freq(each_bigram) # get the frequency of each bigram
        row_results.append(total) # append each author freq total
        #print(total, author)
    
    #dataframe_with_frequencies.append(row_results)
    dataframe_with_frequencies.loc[iter_num+1] = row_results # add each row_results to the dataframe

# Save the dataframe to a CSV file
dataframe_with_frequencies.to_csv("results-bigrams.csv", index=False) # save to csv file