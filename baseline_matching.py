import os
import sys
import json
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stopwords = set(stopwords.words('english'))

def create_output_files(method, dataset, token_dict, obt_dict):
    output_path = "output_" + method + "_" + dataset
    os.mkdir(output_path)
    for doc_name, tokens in token_dict.items():
        doc_name = doc_name[:-1] + '2'
        filepath = os.path.join(output_path, doc_name)
        count = 1
        f= open(filepath,"w+")
        for token in tokens:
            anno_id = token[0]
            anno_string = token[1]
            if method == "jaccard":
                anno_words = set(anno_string.split()).difference(stopwords)
            else:
                anno_words = anno_string.split()
            matching_obt_id = ""
            max_similarity = 0
            for obt_id, info in obt_dict.items():
                for name in info["name"]:
                    if method == "jaccard":
                        words = set(name.split()).difference(stopwords)
                        similarity = len(anno_words.intersection(words))/len(anno_words.union(words))
                        if similarity > max_similarity:
                            max_similarity = similarity
                            matching_obt_id = obt_id
                    else:
                        found_exact = False
                        words = name.split()
                        if anno_string == name and not found_exact:
                            matching_obt_id = obt_id
                            found_exact = True
                        elif anno_string in words and not found_exact:
                            matching_obt_id = obt_id
                        elif name in anno_words and not found_exact:
                            matching_obt_id = obt_id
                    
                if "synonym" in info:
                    for name in info["synonym"]:
                        if method == "jaccard":
                            words = set(name.split()).difference(stopwords)
                            similarity = len(anno_words.intersection(words))/len(anno_words.union(words))
                            if similarity > max_similarity:
                                max_similarity = similarity
                                matching_obt_id = obt_id
                        else:
                            found_exact = False
                            words = name.split()
                            if anno_string == name and not found_exact:
                                matching_obt_id = obt_id
                                found_exact = True
                            elif anno_string in words and not found_exact:
                                matching_obt_id = obt_id
                            elif name in anno_words and not found_exact:
                                matching_obt_id = obt_id
            if matching_obt_id != "":
                f.write("N%d\tOntoBiotope Annotation:%s Referent:%s\n" % (count, anno_id, matching_obt_id))
                count += 1
        f.close()


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

create_output_files("exact", "train", train_token_dict, i_obt_dict)
create_output_files("exact", "dev", dev_token_dict, i_obt_dict)
create_output_files("exact", "test", test_token_dict, i_obt_dict)

create_output_files("jaccard", "train", train_token_dict, i_obt_dict)
create_output_files("jaccard", "dev", dev_token_dict, i_obt_dict)
create_output_files("jaccard", "test", test_token_dict, i_obt_dict)