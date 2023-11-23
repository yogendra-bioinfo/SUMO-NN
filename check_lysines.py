## Author : Yogendra Ramtirtha
## Email : ramtirtha.yogendra@gmail.com
## Prpject : SUMO-NN - a aneural networks based method to predict SUMOylated lysines in human proteins
## This program checks motif status of SUMOylated lysines

input_file1 = open("training_lysines.tsv", "r")
input_data1 = input_file1.readlines()
input_file1.close()
print(len(input_data1))

c1, c2 = 0, 0
for line in input_data1[1:]:
    words = line.split('\t')
    if words[6] == "1":
        c1 += 1
    else:
        c2 += 1
print(c1, c2)

input_file2 = open("testing_lysines.tsv", "r")
input_data2 = input_file2.readlines()
input_file2.close()
print(len(input_data2))

c1, c2 = 0, 0
for line in input_data2[1:]:
    words = line.split('\t')
    if words[6] == "1":
        c1 += 1
    else:
        c2 += 1
print(c1, c2)