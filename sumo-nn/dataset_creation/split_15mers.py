## Author : Yogendra Ramtirtha
## Email : ramtirtha.yogendra@gmail.com
## Prpject : SUMO-NN - a aneural networks based method to predict SUMOylated lysines in human proteins
## 15-mers are assigned to training and testing datasets respectively

input_file = open("all_chosen_15mers.tsv", "r")
input_data = input_file.readlines()
input_file.close()
print(len(input_data))

train_prots = {}
input_file2 = open("training_lysines.tsv", "r")
input_data2 = input_file2.readlines()
input_file2.close()
print(len(input_data2))

for line in input_data2[1:]:
    words = line.strip().split('\t')
    train_prots.setdefault(words[0], {})
    train_prots[words[0]][words[4]] = words[1]
print(len(train_prots))

c1, c2 = 0, 0
outfile1 = open("positive_training_data.tsv", "w")
outfile2 = open("negative_training_data.tsv", "w")
outfile1.writelines(input_data[0])
outfile2.writelines(input_data[0])
for line in input_data[1:]:
    words = line.strip().split('\t')
    if words[0] in train_prots:
        if words[1] in train_prots[words[0]] and train_prots[words[0]][words[1]][18:33] == words[2]:
            c1 += 1
            outfile1.writelines(line)
        else:
            c2 += 1
            outfile2.writelines(line)
outfile1.close()
outfile2.close()
print(c1, c2)

test_prots = {}
input_file3 = open("testing_lysines.tsv", "r")
input_data3 = input_file3.readlines()
input_file3.close()
print(len(input_data3))

for line in input_data3[1:]:
    words = line.strip().split('\t')
    test_prots.setdefault(words[0], {})
    test_prots[words[0]][words[4]] = words[1]
print(len(test_prots))

c3, c4 = 0, 0
outfile3 = open("positive_testing_data.tsv", "w")
outfile4 = open("negative_testing_data.tsv", "w")
outfile3.writelines(input_data[0])
outfile4.writelines(input_data[0])
for line in input_data[1:]:
    words = line.strip().split('\t')
    if words[0] in test_prots:
        if words[1] in test_prots[words[0]] and test_prots[words[0]][words[1]][18:33] == words[2]:
            c3 += 1
            outfile3.writelines(line)
        else:
            c4 += 1
            outfile4.writelines(line)
outfile3.close()
outfile4.close()
print(c3, c4)