import os
import sys
import pickle
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
import re
import nltk


model_filename = 'modelRF.pkl'
vectorizer_filename = 'vector.pkl'
with open(model_filename, 'rb') as model_file, open(vectorizer_filename, 'rb') as vectorizer_file:
    model = pickle.load(model_file)
    vectorizer = pickle.load(vectorizer_file)

def categorize_resume(txt, model, vectorizer):
    text_vector = vectorizer.transform([txt])
    category = model.predict(text_vector)[0]
    return category

if __name__ == "__main__":


    model, vectorizer = load_model_and_vectorizer('modelRF.pkl', 'vector.pkl')

    categorized_resumes = []

    for filename in os.listdir("C:/Users/Abeer/Desktop/project"):
        #print(filename)
        if filename.endswith(".pdf"):
            file_path = os.path.join("C:/Users/Abeer/Desktop/project", filename)
            #print(file_path)
            with open(file_path, 'r') as file:
                txt = file.read().lower()
                txt = txt.replace('{html}', "")  # Removing the string "{html}" from the text
                cleanr = re.compile('<.*?>')  # Creating a regular expression pattern to match and remove HTML tags
                cleantext = re.sub(cleanr, '', txt)
                rem_url = re.sub(r'http\S+', '',
                                 cleantext)  # Removes URLs from the text using a regular expression pattern that matches URLs
                rem_num = re.sub('[0-9]+', '', rem_url)  # Removes numerical digits from the text
                tokenizer = RegexpTokenizer(
                    r'\w+')  # Initializes a tokenizer that splits text into words based on word boundaries
                tokens = tokenizer.tokenize(rem_num)  # Tokenizes the cleaned text, splitting it into individual words
                filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words(
                    'english')]  # Filters out words that are less than 3 characters long and removes common English stopwords.


                category = categorize_resume(txt, model, vectorizer)

                # Move to category folder
                category_folder = os.path.join("C:/Users/Abeer/Desktop/project/pdf", category)
                if not os.path.exists(category_folder):
                    os.mkdir(category_folder)
                os.rename(file_path, os.path.join(category_folder, filename))

                categorized_resumes.append({'filename': filename, 'category': category})

    # Write to CSV
    categorized_df = pd.DataFrame(categorized_resumes)
    categorized_df.to_csv('categorized_resumes.csv', index=False)

    print("Categorization and file moving complete.")
