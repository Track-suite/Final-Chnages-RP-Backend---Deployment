# libraries Used
from asyncio.windows_events import NULL
from flask import Flask, jsonify
from jinja2 import Undefined
from numpy import empty
from rake_nltk import Rake
import pandas as pd
from requests import request
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import string
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

@app.route('/test')
def get_test():
    return "rifan"

@app.route('/searchfoods/<searchValue>/<ingredient>', methods=['GET'])
def search_food(searchValue, ingredient):
    
    value = searchValue
    Ingredient = ingredient.split(',')

    FoodDataset_Base = pd.read_csv('FoodDataSet2.csv')
    FoodDataset_Base = FoodDataset_Base[(FoodDataset_Base["name"].str.upper().str.contains(value.upper()))]

    # Select Coumns
    FoodDataset = FoodDataset_Base[['name', 'ingredients']]

    # clearning Dataset
    FoodDataset['ingredients'] = FoodDataset['ingredients'].str.replace('[^\w\s]', '', regex=True)
    FoodDataset['ingredients'] = FoodDataset['ingredients'].str.replace('[{}]'.format(string.punctuation), '', regex=True)

    # to extract key words from Ingredients to a list
    FoodDataset['Key_words'] = ''

    # use Rake to discard stop words (based on english stopwords from NLTK)
    r = Rake()
    print(r)

    for index, row in FoodDataset.iterrows():
        r.extract_keywords_from_text(
            # to extract key words from Ingredients, default in lower case
            row['ingredients'])
        # to get dictionary with key words and their scores
        key_words_dict_scores = r.get_word_degrees()
        # to assign list of key words to new column
        row['Key_words'] = list(key_words_dict_scores.keys())

    # to combine 4 lists (4 columns) of key words into 1 sentence under Bag_of_words column
    FoodDataset['Bag_of_words'] = ''
    columns = ['Key_words']

    for index, row in FoodDataset.iterrows():
        words = ''
        for col in columns:
            words += ' '.join(row[col]) + ' '
        row['Bag_of_words'] = words

    # strip white spaces infront and behind, replace multiple whitespaces (if any)
    FoodDataset['Bag_of_words'] = FoodDataset['Bag_of_words'].str.strip().str.replace('   ', ' ').str.replace('  ', ' ')

    FoodDataset = FoodDataset[['ingredients', 'Bag_of_words']]

    # to generate the count matrix
    count = CountVectorizer()
    try:
        count_matrix = count.fit_transform(FoodDataset['Bag_of_words'])
    except:
        return "Empty"
    count_matrix

    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    print(cosine_sim)

    # to create a Series for food Positions which can be used as indices (each index is mapped to a food Position)
    indices = pd.Series(FoodDataset['ingredients'])
    print(indices)

    # most simliar input
    from fuzzywuzzy import process
    CollectedDataSet = pd.read_csv('FoodDataSet2.csv')
    Ingredients = FoodDataset['ingredients'].tolist()

    str2Match = Ingredient
    for ing in str2Match:
        idx = process.extractOne(ing, Ingredients)[0]
        print(idx)
    for each in Ingredient:
        FoodDataset_Base = FoodDataset_Base[~FoodDataset_Base['ingredients'].str.upper().str.contains(each.upper())]
        FoodDataset_Base.dropna(inplace=True)

    # print(FoodDataset_Base)
    # print(FoodDataset_Base[['Food_name']])
    # print(FoodDataset_Base.to_json(orient='records'))
    return FoodDataset_Base.to_json(orient='records')

@app.route('/foods/<type>/<ingredient>', methods=['GET'])
def get_food(type, ingredient):
    foodtype = type
    Ingredient = ingredient.split(',')

    FoodDataset_Base = pd.read_csv('FoodDataSet2.csv')

    if foodtype != 'all':
        FoodDataset_Base = FoodDataset_Base[(FoodDataset_Base["tags"].str.upper() == foodtype.upper())]

    # Select Coumns
    FoodDataset = FoodDataset_Base[['name', 'ingredients']]

    # clearning Dataset
    FoodDataset['ingredients'] = FoodDataset['ingredients'].str.replace('[^\w\s]', '', regex=True)
    FoodDataset['ingredients'] = FoodDataset['ingredients'].str.replace('[{}]'.format(string.punctuation), '', regex=True)
    
 
    # to extract key words from Ingredients to a list
    FoodDataset['Key_words'] = ''

    # use Rake to discard stop words (based on english stopwords from NLTK)
    r = Rake()
    print(r)

    for index, row in FoodDataset.iterrows():
        r.extract_keywords_from_text(
            # to extract key words from Ingredients, default in lower case
            row['ingredients'])
        # to get dictionary with key words and their scores
        key_words_dict_scores = r.get_word_degrees()
        # to assign list of key words to new column
        row['Key_words'] = list(key_words_dict_scores.keys())

    # to combine 4 lists (4 columns) of key words into 1 sentence under Bag_of_words column
    FoodDataset['Bag_of_words'] = ''
    columns = ['Key_words']

    for index, row in FoodDataset.iterrows():
        words = ''
        for col in columns:
            words += ' '.join(row[col]) + ' '
        row['Bag_of_words'] = words

    # strip white spaces infront and behind, replace multiple whitespaces (if any)
    FoodDataset['Bag_of_words'] = FoodDataset['Bag_of_words'].str.strip().str.replace('   ', ' ').str.replace('  ', ' ')

    FoodDataset = FoodDataset[['ingredients', 'Bag_of_words']]

    # to generate the count matrix
    count = CountVectorizer()
    count_matrix = count.fit_transform(FoodDataset['Bag_of_words'])
    count_matrix

    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    print(cosine_sim)

    # to create a Series for food Positions which can be used as indices (each index is mapped to a food Position)
    indices = pd.Series(FoodDataset['ingredients'])
    print(indices)
    # print(indices)

    # most simliar input
    from fuzzywuzzy import process
    CollectedDataSet = pd.read_csv('FoodDataSet2.csv')
    Ingredients = FoodDataset['ingredients'].tolist()

    str2Match = Ingredient
    for ing in str2Match:
        idx = process.extractOne(ing, Ingredients)[0]
        print(idx)
    for each in Ingredient:
        FoodDataset_Base = FoodDataset_Base[~FoodDataset_Base['ingredients'].str.upper().str.contains(each.upper())]
        FoodDataset_Base.dropna(inplace=True)

    # print(FoodDataset_Base)
    # print(FoodDataset_Base[['name']])
    # print(FoodDataset_Base.to_json(orient='records'))
    return FoodDataset_Base.to_json(orient='records')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port="3000", debug=True)
