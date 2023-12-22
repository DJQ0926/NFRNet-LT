from config import *
import pandas as pd
import matplotlib.pyplot as plt
import spacy


def count_word_len():
    nlp = spacy.load('en_core_web_sm')  
    text_len = []
    data = pd.read_csv(TRAIN_SAMPLE_PATH)
    for index, row in data.iterrows():
        text, label = row
        doc = nlp(text)  
        words = [token.text for token in doc]
        text_len.append(len(words)) 
    plt.hist(text_len)
    plt.show()
    plt.savefig("/path")
    print(max(text_len))  

    
if __name__ == '__main__':
    count_word_len()