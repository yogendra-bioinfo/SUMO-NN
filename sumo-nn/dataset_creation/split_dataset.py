## Author : Yogendra Ramtirtha
## Email : ramtirtha.yogendra@gmail.com
## Prpject : SUMO-NN - a aneural networks based method to predict SUMOylated lysines in human proteins
## This program randomly shuffles and splits training and testing datasets in 80:20 ratio

import random

input_file1 = open("chosen_lysines.tsv", "r")
input_data1 = input_file1.readlines()
input_file1.close()
print(len(input_data1))

input_file2 = open("chosen_lysines_per_protein.tsv", "r")
input_data2 = input_file2.readlines()
input_file2.close()
print(len(input_data2))

full_list = []
for entry in input_data2[1:]:
    full_list.append(entry)
print(len(full_list))

random.shuffle(full_list)

train_prots, test_prots = [], []

outfile1 = open("training_lysines_per_protein.tsv", "w")
outfile2 = open("testing_lysines_per_protein.tsv", "w")
outfile1.writelines(input_data2[0])
outfile2.writelines(input_data2[0])
#for (i,line) in enumerate(input_data2[1:]):
for (i,line) in enumerate(full_list):
    words = line.split('\t')
    if i < 3000:
        train_prots.append(words[0])
        outfile1.writelines(line)
    else:
        test_prots.append(words[0])
        outfile2.writelines(line)
outfile1.close()
outfile2.close()
print(len(train_prots), len(test_prots))

c = 0
outfile3 = open("training_lysines.tsv", "w")
outfile3.writelines(input_data1[0])
for line in input_data1[1:]:
    words = line.split('\t')
    if words[0] in train_prots:
        outfile3.writelines(line)
        c += 1
outfile3.close()
print(c)

c = 0
outfile4 = open("testing_lysines.tsv", "w")
outfile4.writelines(input_data1[0])
for line in input_data1[1:]:
    words = line.split('\t')
    if words[0] in test_prots:
        outfile4.writelines(line)
        c += 1
outfile4.close()
print(c)