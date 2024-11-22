# Sentiment Index Data Records
This repository demonstrates how to calculate sentiment index from text data using the KnuSentiLex sentiment dictionary and text review datasets. The project combines a tokenized review dataset and a sentiment lexicon to produce sentiment index for text data.

## Sentiment Index Calculating
The sentiment dictionary is loaded as a Pandas DataFrame using `pd.read_csv()`. It maps Korean words or phrases (`ngram`) to their sentiment scores (`max.value`).

The tokenized text data is also loaded as a Pandas DataFrame. Each entry in the `tokenized` column contains a list of tokens. The code processes this column to convert the tokenized list into a whitespace-separated string for analysis.

The function `calculate_sentiment()` computes sentiment indices:
- It splits a text string into tokens.
- For each token, it checks if it exists in the sentiment dictionary.
- If a match is found, the corresponding sentiment score from `max.value` is added to a cumulative total.

The sentiment calculation is applied to all rows in the tokenized text dataset using the Pandas `.apply()` method. A new column, `sentiment_score`, is added to the dataset to store the calculated scores.

```python
import pandas as pd

# Load sentiment dictionary
sentiment_dict_path = '/Users/dictionary directory'  # Update with the correct path to the KnuSentiLex dictionary
sentiment_dict = pd.read_csv(sentiment_dict_path)

# Load tokenized text data
tokenized_data_path = '/Users/tokenizedReview directory'  # Update with the correct path to the tokenized review data
tokenized_data = pd.read_csv(tokenized_data_path)

# Convert tokenized text to string
tokenized_data['tokenized'] = tokenized_data['tokenized'].apply(lambda x: ' '.join(eval(x)))  # Convert list of tokens to string

# Calculate sentiment scores
def calculate_sentiment(text):
    tokens = text.split()  # Assuming tokens are separated by whitespace
    sentiment_score = 0
    for token in tokens:
        if token in sentiment_dict['ngram'].values:  # Check if the token exists in the sentiment dictionary
            sentiment_score += sentiment_dict[sentiment_dict['ngram'] == token]['max.value'].values[0]  # Add the sentiment score
    return sentiment_score

# Apply sentiment calculation to tokenized text data
tokenized_data['sentiment_score'] = tokenized_data['tokenized'].apply(calculate_sentiment)

# Save tokenized text data with sentiment scores to CSV
tokenized_data.to_csv(tokenized_data_path, index=False)

# Print tokenized text data with sentiment scores
print(tokenized_data)
```
