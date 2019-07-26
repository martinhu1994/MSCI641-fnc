import re
import string
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize 


PUNCTUATION = string.punctuation
NUM_REG_EXP = r'^[0-9]+[,.][0-9]+'
ACRONYM_REG_EXP = r'^([a-z][.])([a-z][.])+$'
ABBREVIATION = ["'ve", "n't", "'re", "'s", "'d"]
CLASSIFICATION = ['unrelated', 'discuss', 'agree', 'disagree']


def double_check(token):
    if token.startswith('-'):
        token = token[1:]
    return token


def clean_token(token):
    if token in ABBREVIATION:
        return [token]
    
    # u.s. || d.c.
    if not re.search(ACRONYM_REG_EXP, token) is None:
        return [token]
    
    new_tokens = []
    start = 0

    # 10,000 || 1.69 || 40,000ft
    match = re.search(NUM_REG_EXP, token)
    if not match is None:
        span = match.span()
        if span[1] == len(token):
            return [token]
        else:
            new_tokens.append(token[:span[1]])
            token = token[span[1]:]

    for i in range(len(token)):
        char = token[i]
        if char.isalpha() or char.isnumeric():
            continue
        if i - start != 0:
            new_tokens.append(double_check(token[start:i]))
        start = i + 1
    
    if start < len(token):
        new_tokens.append(double_check(token[start:]))

    return new_tokens


def process_tokens(tokens):
    new_article = []
    for token in tokens:
        # valid word
        if token.isalpha() or token.isnumeric():
            new_article.append(token)
        # invalid word case 1: single punctuation
        elif token in PUNCTUATION:
            continue
        # invalid word case 2: single character(special symbol || one letter)
        elif len(token) == 1:
            continue
        # need further text processing
        else:
            new_tokens = clean_token(token)
            for t in new_tokens:
                if len(t) == 1 and not t.isnumeric():
                    continue
                new_article.append(t)
    return new_article


def buidl_mapping(articles, ids):
    id_article_map = dict()
    for i in range(len(ids)): 
        tokens = word_tokenize(articles[i].lower())
        new_article = ' '.join(process_tokens(tokens))
        id_article_map[int(ids[i])] = new_article
    return id_article_map


def generate_new_file(headlines, ids, stances, id_article_map, precossed_file):
    lines = [['Headline', 'Article', 'Stance']]
    for i in range(len(ids)):
        tokens = word_tokenize(headlines[i].lower())
        new_headline = ' '.join(process_tokens(tokens))
        line = []
        line.append(new_headline)
        line.append(id_article_map[int(ids[i])])
        line.append(stances[i])
        lines.append(line)
    np.savetxt('data/' + precossed_file, lines, delimiter=",", fmt='\"%s\"')


def process_raw(bodies_file, stances_file, precossed_file):
    data = pd.read_csv('data/' + bodies_file)
    articles = data['articleBody']
    ids = data['Body ID']

    id_article_map = buidl_mapping(articles, ids)

    data = pd.read_csv('data/' + stances_file)
    headlines = data['Headline']
    ids = data['Body ID']
    stances = data['Stance']

    generate_new_file(headlines, ids, stances, id_article_map, precossed_file)

    df = pd.read_csv('data/' + precossed_file)
    print(df.head())


def main():
    print("Tokenizing Traning Data ...")
    process_raw("train_bodies.csv", "train_stances.csv", "processed_data.csv")
    print()
    print("Tokenizing Test Data ...")
    process_raw("competition_test_bodies.csv", "competition_test_stances.csv", "processed_test.csv")


if __name__ == '__main__':
    main()

