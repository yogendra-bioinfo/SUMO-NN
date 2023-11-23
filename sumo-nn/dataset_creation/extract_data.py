## Author : Yogendra Ramtirtha
## Email : ramtirtha.yogendra@gmail.com
## Prpject : SUMO-NN - a aneural networks based method to predict SUMOylated lysines in human proteins
## This program extracts list of human proteins and the SUMOylates lysines in them from supplementary data of Hendriks, et al, 2018

from collections import OrderedDict

input_file = open("41467_2018_4957_MOESM6_ESM-Hendriks_Data_S3.tsv", "r")
input_data = input_file.readlines()
input_file.close()
print(len(input_data))

header = []
for i,j in enumerate(input_data[1].strip().split('\t')):
    print(i, j)
    header.append(j)
print(len(header))

new_header = [header[0], header[1], header[2], header[3], header[4], header[18], header[31], header[34]]

c = 0
map_dict = OrderedDict()
outfile = open("all_available_lysines.tsv", "w")
outfile.writelines("\t".join(new_header) + "\n")
for line in input_data[2:]:
    words = line.strip().split('\t')
    #print(words)
    if (words[18] != "" and int(words[18]) >= 2) or (words[34] != "" and int(words[34]) >= 2):
        c += 1
        new_line = [words[0], words[1], words[2], words[3], words[4], words[18], words[31], words[34]]
        outfile.writelines("\t".join(new_line) + "\n")
        map_dict.setdefault(words[0], [])
        map_dict[words[0]].append(words[4])
outfile.close()
print(c)
print(len(map_dict))

c = 0
for i in map_dict:
    c += len(map_dict[i])
print(c)

outfile = open("available_lysines_per_protein.tsv", "w")
outfile.writelines("Protein" + "\t" + "Lysines_list" + "\n")
for i in map_dict:
    outfile.writelines(i + "\t" + ";".join(map_dict[i]) + "\n")
outfile.close()

outfile = open("protein_sequences_to_be_downloaded.txt", "w")
for i in map_dict:
    outfile.writelines(i + "\n")
outfile.close()