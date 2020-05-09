import re
import json

input = open("OntoBiotope_BioNLP-OST-2019.obo","r")
output = open("dict.json","w")
myDict = {}
keys = ["name","is_a","synonym"]
for line in input.readlines():
    line =line.strip()
    if line[:4] == "id: ":
        current_id = line[4:]
        myDict[current_id] = {}
    else:
        for key in keys:
            if line[:len(key)] == key:
                if key == "name":
                    value = line[6:]
                elif key == "is_a":
                    line = line[6:]
                    id = re.search('(.*)(?= !)', line).group(0)
                    is_a = re.search('(?<=! )(.*)', line).group(0)
                    value=(id,is_a)
                else:
                    value = re.search('(?<=\")(.*)(?=\")', line).group(0)

                if key != "name" and key not in myDict[current_id]:
                    myDict[current_id][key]=[value]
                elif key != "name":
                    myDict[current_id][key].append(value)
                else:
                    myDict[current_id][key]=value                   
json.dump(myDict, output,indent=4, sort_keys=True)
