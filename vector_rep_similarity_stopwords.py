from gensim.models import KeyedVectors
from scipy import spatial
import numpy as np
import json
import os
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stopwords = stopwords.words('english')

model = KeyedVectors.load_word2vec_format('wikipedia-pubmed-and-PMC-w2v.bin', binary=True )

if(os.path.isfile("dict.json")):
    with open("dict.json") as jf:
        i_obt_dict = json.load(jf)
else:
    print("OBT dataset dictionary not found.")
    exit()

if(os.path.isfile("tokens_train.json")):
    with open("tokens_train.json") as jf:
        train_token_dict = json.load(jf)
else:
    print("A1 tokens not found.")
    exit()
if(os.path.isfile("tokens_dev.json")):
    with open("tokens_dev.json") as jf:
        dev_token_dict = json.load(jf)
else:
    print("A1 tokens not found.")
    exit()
if(os.path.isfile("tokens_test.json")):
    with open("tokens_test.json") as jf:
        test_token_dict = json.load(jf)
else:
    print("A1 tokens not found.")
    exit()

def create_obt_wordvec(obt_dict):
    obt_wordvec = dict()
    for obt_id, info in obt_dict.items():
        obt_wordvec[obt_id]=[]
        words = info["name"].split()
        word_vec = np.array([model[word] for word in words if word in model and word not in stopwords])
        if len(word_vec) != 0:
            mean_word = word_vec.mean(axis=0)
            obt_wordvec[obt_id].append(mean_word)
        if "synonym" in info:
            for name in info["synonym"]:
                words = name.split()
                word_vec = np.array([model[word] for word in words if word in model and word not in stopwords])
                if len(word_vec) == 0:
                    continue
                mean_word = word_vec.mean(axis=0)
                obt_wordvec[obt_id].append(mean_word)
    return obt_wordvec

obt_wordvec = create_obt_wordvec(i_obt_dict)

def calculate_cosine_similarity(token_dict,obt_wordvec,dataset):
    output_path = "output_word2vec_stopwords_" + dataset
    os.mkdir(output_path)
    for file_name, list in token_dict.items():
        count=1
        doc_name = file_name[:-1] + '2'
        filepath = os.path.join(output_path, doc_name)
        f= open(filepath,"w+")
        for tuple in list:
            values = [model[x] for x in tuple[1].split() if x in model]
            if len(values) == 0:
                continue
            mean = np.array(values).mean(axis=0)

            max_similarity = 0
            matching_obt_id = 0

            for obt_id, wordvec_list in obt_wordvec.items():
                if len(wordvec_list) == 0:
                    continue
                # for mean_term in wordvec_list: 
                #     cosine_sim = 1 - spatial.distance.cosine(mean, mean_term)
                similarity = np.max(np.array([1 - spatial.distance.cosine(mean, mean_term) for mean_term in wordvec_list]),axis=0)
                if similarity > max_similarity:
                    max_similarity = similarity
                    matching_obt_id = obt_id

            if matching_obt_id != 0:
                f.write("N%d\tOntoBiotope Annotation:%s Referent:%s\n" % (count, tuple[0], matching_obt_id))
                count += 1
        f.close()

calculate_cosine_similarity(dev_token_dict,obt_wordvec,"dev")