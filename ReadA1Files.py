import glob
import json

a1_filenames = glob.glob("*.a1")
dictionary = {}
count = 0
for filename in a1_filenames:
    dictionary[filename] = {}
    list = []
    for line in open(filename, "r").readlines():
        if line.split()[1].__eq__("Habitat"):
            list.append([line.split()[0], " ".join(line.split()[4:])])
    if list is not []:
        dictionary[filename] = list

with open('tokens.json', 'w') as outfile:
    json.dump(dictionary, outfile)
