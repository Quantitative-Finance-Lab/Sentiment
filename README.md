<img src="https://img.shields.io/badge/Google Colab-F9ABOO?style=for-the-badge&logo=Google Colab&logoColor=white" link='https://colab.google/'> <img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white">  

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

## Spatial Interpolation
Spatial interpolation step can be utilized to remedy the uneven spatial distribution of GSV images.   
To implement the spatial interpolation method, refer to the sample data file named *'Data.csv'* and *Street Greenness.csv*.    
The columns required to effectively manage the green index are as follows:   

*Data.csv*
- x: Longitude in the Cartesian coordinate system of the transaction point
- y: Latitude in the Cartesian coordinate system of the transaction point
   
*Street Greenness.csv*
- Longitude: Longitude of GSV image
- Latitude: Latitude of GSV image
- Green Index: Calculated street greenness

Spatial interpolation requires the distance between two objects based on longitude and latitude. It can be obtained by using haversine formula as follows:

```math
d_{\text{haversine}} = 2 \times R \times \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta \text{lat}}{2}\right) + \cos(\text{lat}_p) \cos(\text{lat}_g) \sin^2\left(\frac{\Delta \text{lng}}{2}\right)}\right)
```

   
<p align="center">
  <img src = "/README_image/spatial interpolation.png" width = "60%"> <br>
  Figure 3. Graphical description of spatial interpolation.
</p>   

The following code uses above mathematical form and aggregates the green index with 50 images closest to the transaction point. The final result file is in *Green Index_Spatial Interpolation_bs.csv*.
```python
import pandas as pd
import pandas as pd
from haversine import haversine

area = ['bs', 'dg', 'dj', 'gw']

for i in range(0, len(area)):
    name = area[i]

    df = pd.read_excel(f'Delentropy\df_{name}.xlsx')
    df['Delentropy'] = ''
    delentropy = pd.read_csv(f'Delentropy\del_{name}.csv')

    df['index'] = df['index'].astype(str)
    delentropy['index'] = delentropy['index'].astype(str)

    del_df = pd.merge(df, delentropy, on=['index'], how ='left')
    del_df.drop(columns = ['Delentropy'], inplace=True)
    del_df.to_excel(f'Delentropy\df_{name}_del.xlsx', index=False)

    ## Spatial Interpolation
    del_df_1 = del_df[del_df['delentropy'].isna()].reset_index()
    dff = del_df[['Latitude', 'Longitude', 'delentropy']].copy()
    dff = dff[dff['delentropy'].notna()].drop_duplicates().reset_index(drop=True)

    Aggregated_Entropy = []
    Aggregated_Entropy_Distance = []
    del_df['delentropy_d'] = ''

    a = 0

    for y, x, ind in zip(del_df_1['Latitude'], del_df_1['Longitude'], del_df_1.index):
        distance = []

        for en_y, en_x, hgvi in zip(dff['Latitude'], dff['Longitude'], dff['delentropy']):
            dis = haversine([y,x], [en_y, en_x], unit='km')
            distance.append([x,y,en_x,en_y,dis,hgvi])
        dis_df = pd.DataFrame(distance)
        dis_df.columns = ['x','y','en_x','en_y','distance','HGVI']
        dis_df = dis_df.sort_values('distance', ascending=True)

        # Extract the 100 nearest green indices
        dis_df_100 = dis_df.iloc[:100]

        mean_hgvi_100 = dis_df_100['HGVI'].mean()
        mean_dis_100 = dis_df_100['distance'].mean()

        Aggregated_Entropy.append(mean_hgvi_100)
        Aggregated_Entropy_Distance.append(mean_dis_100)

        a += 1

        print(a, '/', len(del_df_1))

    del_df_1['delntropy'] = Aggregated_Entropy
    del_df_1['delentropy_d'] = Aggregated_Entropy_Distance

    for i in range(0,len(del_df_1)):
        del_df['delentropy'][del_df_1['level_0'][i]] = Aggregated_Entropy[i]
        del_df['delentropy_d'][del_df_1['level_0'][i]] = Aggregated_Entropy_Distance[i]

    del_df.to_csv(f'Delentropy\spatial_interpolation_{name}.csv',index=False,encoding='utf-8-sig')
```
Through this process, we can get the green index for all points of transaction and all information of hedonic variables including green index is in *Hedonic Dataset.xlsx*.
