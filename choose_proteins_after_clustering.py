## Author : Yogendra Ramtirtha
## Email : ramtirtha.yogendra@gmail.com
## Prpject : SUMO-NN - a aneural networks based method to predict SUMOylated lysines in human proteins
## This program analyzes protein list obtained after clustering by Linclust program in MMseqs2 software suite

input_file1 = open("proteins_available_after_clustering.fasta", "r")
input_data1 = input_file1.readlines()
input_file1.close()
print(len(input_data1))

prot_list = []
for line in input_data1:
    if line[0] == ">":
        words = line.split('|')
        prot_list.append(words[1])
print(len(prot_list))

input_file2 = open("all_available_lysines.tsv", "r")
input_data2 = input_file2.readlines()
input_file2.close()
print(len(input_data2))

input_file3 = open("available_lysines_per_protein.tsv", "r")
input_data3 = input_file3.readlines()
input_file3.close()
print(len(input_data3))

c = 0
outfile1 = open("chosen_lysines.tsv", "w")
outfile1.writelines(input_data2[0])
for line in input_data2[1:]:
    words = line.split('\t')
    if words[0] in prot_list:
        c += 1
        outfile1.writelines(line)
outfile1.close()
print(c)

c = 0
outfile2 = open("chosen_lysines_per_protein.tsv", "w")
outfile2.writelines(input_data3[0])
for line in input_data3[1:]:
    words = line.split('\t')
    if words[0] in prot_list:
        c += 1
        outfile2.writelines(line)
outfile2.close()
print(c)