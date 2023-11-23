## Author : Yogendra Ramtirtha
## Email : ramtirtha.yogendra@gmail.com
## Prpject : SUMO-NN - a aneural networks based method to predict SUMOylated lysines in human proteins
## This program analyzes protein sequences downloaded from UniProt database

from collections import OrderedDict

input_file = open("idmapping_2023_09_23.fasta",  "r")
input_data = input_file.readlines()
input_file.close()
print(len(input_data))

map_dict = OrderedDict()

c = 0
for line in input_data:
    if line[0] == ">":
        c += 1
print(c)

for index1 in range(len(input_data)):
    line = input_data[index1]
    if line[0] == ">":
        unp_id = line.split('|')[1]
        map_dict.setdefault(unp_id, {})
        map_dict[unp_id]["header"] = line
        map_dict[unp_id]["sequence"] = ""
        for index2 in range(index1 + 1, len(input_data)):
            next_line = input_data[index2]
            if next_line[0] == ">":
                break
            else:
                map_dict[unp_id]["sequence"] += next_line
    else:
        continue
print(len(map_dict))

input_file2 = open("protein_sequences_to_be_downloaded.txt", "r")
input_data2 = input_file2.readlines()
input_file2.close()
print(len(input_data2))

c1, c2 = 0, 0
outfile = open("input_list_for_mmseq2.fasta", "w")
for line in input_data2:
    protein = line.strip()
    if protein in map_dict:
        outfile.writelines(map_dict[protein]["header"])
        outfile.writelines(map_dict[protein]["sequence"])
        c1 += 1
    else:
        c2 += 1
        print(protein)
outfile.close()
print(c1, c2)