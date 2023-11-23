## Author : Yogendra Ramtirtha
## Email : ramtirtha.yogendra@gmail.com
## Prpject : SUMO-NN - a aneural networks based method to predict SUMOylated lysines in human proteins
## This program extracts 15-mers centered on every lysine residue of all proteins used in this study
## 15-mers are extracted by including 7 amino acid residues before and 7 amino acid residues, while lysien is at centre

from collections import OrderedDict

input_file = open("proteins_available_after_clustering.fasta",  "r")
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

window_size = 7
c = 0
all_15mers = []
outfile = open("all_chosen_15mers.tsv", "w")
outfile.writelines("\t".join(["Protein_UniProt_ID", "Lysine_Position", "15mer_centered_on_lysines", "Protein_Name"]) + "\n")
for prot_id in map_dict:
    for (index, aa) in enumerate(map_dict[prot_id]["sequence"]):
        if aa == "K":
            seq_15mer = ""
            window_start = index - window_size
            window_end = index + window_size + 1
            if window_start < 0:
                window_start *= -1
                overhang = "".join(["_" for i in range(window_start)])
                seq_15mer = overhang + map_dict[prot_id]["sequence"][0 : window_end]
                all_15mers.append(seq_15mer)
                outfile.writelines("\t".join([prot_id, str(index + 1), seq_15mer, map_dict[prot_id]["header"]]) + "\n")
            elif window_end > (len(map_dict[prot_id]["sequence"]) - 1):
                difference = window_end - len(map_dict[prot_id]["sequence"])
                overhang = "".join(["_" for i in range(difference)])
                seq_15mer = map_dict[prot_id]["sequence"][window_start : len(map_dict[prot_id]["sequence"])] + overhang
                all_15mers.append(seq_15mer)
                outfile.writelines("\t".join([prot_id, str(index + 1), seq_15mer, map_dict[prot_id]["header"]]) + "\n")
            else:
                seq_15mer = map_dict[prot_id]["sequence"][window_start : window_end]
                all_15mers.append(seq_15mer)
                outfile.writelines("\t".join([prot_id, str(index + 1), seq_15mer, map_dict[prot_id]["header"]]) + "\n")
            c += 1
outfile.close()
print(c)
print(len(all_15mers))

c = 0
for i in all_15mers:
    if len(i) != 15:
        print(i)