import os
import sys
import json

stopwords = {"ourselves", "hers", "between", "yourself", "but", "again", "there", "about", "once", "during", "out", "very", "having", "with", "they", "own", "an", "be", "some", "for", "do", "its", "yours", "such", "into", "of", "most", "itself", "other", "off", "is", "s", "am", "or", "who", "as", "from", "him", "each", "the", "themselves", "until", "below", "are", "we", "these", "your", "his", "through", "don", "nor", "me", "were", "her", "more", "himself", "this", "down", "should", "our", "their", "while", "above", "both", "up", "to", "ours", "had", "she", "all", "no", "when", "at", "any", "before", "them", "same", "and", "been", "have", "in", "will", "on", "does", "yourselves", "then", "that", "because", "what", "over", "why", "so", "can", "did", "not", "now", "under", "he", "you", "herself", "has", "just", "where", "too", "only", "myself", "which", "those", "i", "after", "few", "whom", "t", "being", "if", "theirs", "my", "against", "a", "by", "doing", "it", "how", "further", "was", "here", "than"}

if(os.path.isfile("dict.json")):
    with open("dict.json") as jf:
        obt_dict = json.load(jf)
else:
    print("OBT dataset dictionary not found.")
    exit()
if(os.path.isfile("tokens.json")):
    with open("tokens.json") as jf:
        token_dict = json.load(jf)
else:
    print("A1 tokens not found.")
    exit()

output_path = sys.argv[1]
os.mkdir(output_path)
for doc_name, tokens in token_dict.items():
    doc_name = doc_name[:-1] + '2'
    filepath = os.path.join(output_path, doc_name)
    count = 1
    f= open(filepath,"w+")
    for token in tokens:
        anno_id = token[0]
        anno_string = token[1]
        anno_words = set(anno_string.split()).difference(stopwords)
        #anno_words = anno_string.split()
        matching_obt_id = ""
        max_similarity = 0
        for obt_id, info in obt_dict.items():
            found_exact = False
            for name in info["name"]:
                words = set(name.split()).difference(stopwords)
                similarity = len(anno_words.intersection(words))/len(anno_words.union(words))
                if similarity > max_similarity:
                    max_similarity = similarity
                    matching_obt_id = obt_id
                '''
                words = name.split()
                if anno_string == name and not found_exact:
                    matching_obt_id = obt_id
                    found_exact = True
                elif anno_string in words and not found_exact:
                    matching_obt_id = obt_id
                elif name in anno_words and not found_exact:
                    matching_obt_id = obt_id
                '''
        if matching_obt_id != "":
            f.write("N%d\tOntoBiotope Annotation:%s Referent:%s\n" % (count, anno_id, matching_obt_id))
            count += 1
    f.close()
