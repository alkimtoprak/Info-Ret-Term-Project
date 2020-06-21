from gensim.models import KeyedVectors
from scipy import spatial
import numpy as np
import json
import os
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stopwords = stopwords.words('english')

model = KeyedVectors.load_word2vec_format('wikipedia-pubmed-and-PMC-w2v.bin', binary=True)

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
    mean_obt_wordvec = dict()
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
        mean_vec = np.array([x for x in obt_wordvec[obt_id]])
        if len(mean_vec) == 0:
            continue 
        mean_obt_wordvec[obt_id] = mean_vec.mean(axis=0)
    return obt_wordvec, mean_obt_wordvec

obt_wordvec, mean_obt_wordvec = create_obt_wordvec(i_obt_dict)

def adjust_obt(mean_obt_wordvec, obt_dict, obt_wordvec):
    is_a_dict = dict()
    parent_adjusted_wordvec = dict()
    child_adjusted_wordvec = dict()
    for obt_id_is_a, info in obt_dict.items():
        arr = np.array([mean_obt_wordvec[obt_id] for obt_id, info in obt_dict.items() 
            if "is_a" in info and info["is_a"][0][0] == obt_id_is_a and obt_id in mean_obt_wordvec])
        if len(arr) == 0:
            continue
        is_a_dict[obt_id_is_a] = arr.mean(axis=0)
    for obt_id, wordvec_list in obt_wordvec.items():
        parent_adjusted_wordvec[obt_id]=[]
        if obt_id in is_a_dict:
            for wordvec in wordvec_list:
                parent_adjusted_wordvec[obt_id].append(np.add(0.75 * wordvec, 0.25 * is_a_dict[obt_id]))
        else:
            parent_adjusted_wordvec[obt_id] = obt_wordvec[obt_id]
    for obt_id_child, info in obt_dict.items():
        if "is_a" in info and info["is_a"][0][0] in is_a_dict:
            child_adjusted_wordvec[obt_id_child] = []
            is_a_vector = is_a_dict[info["is_a"][0][0]]
            for wordvec in parent_adjusted_wordvec[obt_id_child]:
                child_adjusted_wordvec[obt_id_child].append(np.add(0.85 * wordvec, 0.15 * is_a_vector))
        else:
            child_adjusted_wordvec[obt_id] = parent_adjusted_wordvec[obt_id]
    #print(adjusted_wordvec["OBT:000129"])
    return child_adjusted_wordvec

is_a_adjusted_wordvec = adjust_obt(mean_obt_wordvec,i_obt_dict, obt_wordvec)

def calculate_cosine_similarity(token_dict,obt_wordvec,dataset,obt_dict):
    output_path = "output_word2vec_jaccard0.7_" + dataset
    os.mkdir(output_path)
    for file_name, list in token_dict.items():  
        last_vec = np.array([])
        count=1
        doc_name = file_name[:-1] + '2'
        filepath = os.path.join(output_path, doc_name)
        f= open(filepath,"w+")
        for tuple in list:
            anno_words = set(tuple[1].split()).difference(stopwords)
            values = [model[x] for x in tuple[1].split() if x in model and x not in stopwords]
            if len(values) == 0:
                continue
            mean = np.array(values).mean(axis=0)
            if last_vec.size == 0:
                last_vec = mean
            else:
                last_vec = np.vstack([last_vec, mean])
            #if last_vec.size != 0:
                #mean_temp = mean
            history = last_vec.mean(axis=0)
            mean = np.add(0.95 * mean, 0.05 * history)
                #last_vec = np.add(0.5 * mean_temp, 0.5 * last_vec)
            max_similarity = 0
            matching_obt_id = 0

            for obt_id, info in obt_dict.items():
                for name in info["name"]:
                    words = set(name.split()).difference(stopwords)
                    similarity = len(anno_words.intersection(words))/len(anno_words.union(words))
                    if similarity > max_similarity:
                        max_similarity = similarity
                        matching_obt_id = obt_id
                if "synonym" in info:
                    for name in info["synonym"]:
                        words = set(name.split()).difference(stopwords)
                        similarity = len(anno_words.intersection(words))/len(anno_words.union(words))
                        if similarity > max_similarity:
                            max_similarity = similarity
                            matching_obt_id = obt_id
            if max_similarity <= 0.7:
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
            #last_vec = np.array([x for x in obt_wordvec[matching_obt_id]]).mean(axis=0)
            if matching_obt_id != 0:
                f.write("N%d\tOntoBiotope Annotation:%s Referent:%s\n" % (count, tuple[0], matching_obt_id))
                count += 1
        f.close()

calculate_cosine_similarity(test_token_dict,is_a_adjusted_wordvec,"test", i_obt_dict)