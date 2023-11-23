## Author : Yogendra Ramtirtha
## Email : ramtirtha.yogendra@gmail.com
## Prpject : SUMO-NN - a aneural networks based method to predict SUMOylated lysines in human proteins
## This program saves protein sequences of training and testing datasets

from collections import OrderedDict

input_file = open("idmapping_2023_09_23.fasta",  "r")
input_data = input_file.readlines()
input_file.close()
print(len(input_data))

input_file2 = open("training_lysines_per_protein.tsv",  "r")
input_data2 = input_file2.readlines()
input_file2.close()
print(len(input_data2))

input_file3 = open("testing_lysines_per_protein.tsv",  "r")
input_data3 = input_file3.readlines()
input_file3.close()
print(len(input_data3))

map_dict = OrderedDict()

c = 0
for line in input_data:
    if line[0] == ">":
        c += 1
print(c)

for index1 in range(len(input_data)):
    line = input_data[index1]
    line = line.strip()
    if line[0] == ">":
        unp_id = line.split('|')[1]
        map_dict.setdefault(unp_id, {})
        map_dict[unp_id]["header"] = line
        map_dict[unp_id]["sequence"] = ""
        for index2 in range(index1 + 1, len(input_data)):
            next_line = input_data[index2]
            next_line = next_line.strip()
            if next_line[0] == ">":
                break
            else:
                map_dict[unp_id]["sequence"] += next_line
    else:
        continue
print(len(map_dict))

train_prots, test_prots = [], []

for line in input_data2[1:]:
    words = line.strip().split('\t')
    train_prots.append(words[0])
print(len(train_prots))

for line in input_data3[1:]:
    words = line.strip().split('\t')
    test_prots.append(words[0])
print(len(test_prots))

c = 0
outfile2 = open("training_proteins.fasta", "w")
for prot in train_prots:
    outfile2.writelines(map_dict[prot]["header"] + "\n")
    outfile2.writelines(map_dict[prot]["sequence"] + "\n")
    c += 1
outfile2.close()
print(c)

c = 0
outfile3 = open("testing_proteins.fasta", "w")
for prot in test_prots:
    outfile3.writelines(map_dict[prot]["header"] + "\n")
    outfile3.writelines(map_dict[prot]["sequence"] + "\n")
    c += 1
outfile3.close()
print(c)