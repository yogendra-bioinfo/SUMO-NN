## Author : Yogendra Ramtirtha
## Email : ramtirtha.yogendra@gmail.com
## Prpject : SUMO-NN - a aneural networks based method to predict SUMOylated lysines in human proteins

## Here, PyTorch library and its associated functionalities are imported
import torch
torch.manual_seed(1)
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
from torch.autograd import Variable
import torch.nn.functional as F
import time

## This function takes predictions and class labels and calculates confusion matrix
## Here, tp = true positives, fp = false positives, tn = true negatives, fn = false negatives
def get_confusion_matrix(prediction, target):
    count_dict = {}
    count_dict[0] = {}
    count_dict[1] = {}
    count_dict[0][0] = 0
    count_dict[0][1] = 0
    count_dict[1][0] = 0
    count_dict[1][1] = 0
    for (value1, value2) in zip(prediction, target):
        count_dict[value1.item()][value2.item()] += 1
    return (count_dict[1][1], count_dict[1][0], count_dict[0][0], count_dict[0][1])


## This function calculates f1 score
def get_f1_score(tp, fp, tn, fn):
    f1 = float(tp)/(tp + 0.5 * (fp + fn))
    return f1

## This function tokenizes 15-mers into position specific amino acid occurrences
def tokenize_15mer(input_15mer):
    tokens = []
    for (index, aa) in enumerate(input_15mer):
        if aa != "_" and aa != "X":
            tokens.append(str(index) + aa)
        else:
            tokens.append("<unk>")
    k_index = tokens.index("7K")
    del tokens[k_index]
    return tokens

## All the functions and class given below are important for training and testing the SUMO-NN model
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenize_15mer(text)

def collate_batch(batch):
    label_list, text_list = [], []
    for _label, _text in batch:
        label_list.append(label_pipeline(_label))
        processed_text = text_pipeline(_text)
        text_list.append(processed_text)
    label_list = torch.tensor(label_list, dtype=torch.int64)
    text_list = torch.tensor(text_list, dtype = torch.long)
    return label_list.to(device), text_list.to(device)

class SUMO_NN_Model(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(SUMO_NN_Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, vocab_size,batch_first = True)
        self.output = nn.Linear(vocab_size, 1)

    def forward(self, text):
        embedded = self.embedding(text)
        result, _ = self.rnn(embedded)
        last_hidden = result[:, -1, :]
        logits = self.output(last_hidden)
        return logits


def train(dataloader):
    model.train()
    tp, fp, tn, fn = 0, 0, 0, 0
    total_f1 = 0.0
    start_time = time.time()

    for idx, (label, text) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text)
        predicted_prob = (predicted_label > 0.0).float()
        label = label.unsqueeze(1)
        loss = criterion(predicted_label, label.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        conf_mat = get_confusion_matrix(predicted_prob, label)
        tp += conf_mat[0]
        fp += conf_mat[1]
        tn += conf_mat[2]
        fn += conf_mat[3]
    total_f1 = get_f1_score(tp, fp, tn, fn)
    elapsed = time.time() - start_time
    print(
        "| epoch {:3d} | {:5d}/{:5d} batches "
        "| tp, fp, tn, fn, f1_score ={:5d}, {:5d}, {:5d}, {:5d}, {:0.3f}".format(
            epoch, len(dataloader), len(dataloader), tp, fp, tn, fn, total_f1
        )
    )


def evaluate(dataloader):
    model.eval()
    tp, fp, tn, fn = 0, 0, 0, 0
    total_f1 = 0.0
    total_loss = 0.0

    with torch.no_grad():
        for idx, (label, text) in enumerate(dataloader):
            predicted_label = model(text)
            predicted_prob = (predicted_label > 0.0).float()
            label = label.unsqueeze(1)
            loss = criterion(predicted_label, label.float())
            conf_mat = get_confusion_matrix(predicted_prob, label)
            tp += conf_mat[0]
            fp += conf_mat[1]
            tn += conf_mat[2]
            fn += conf_mat[3]
            total_loss += loss.item()
    total_f1 = get_f1_score(tp, fp, tn, fn)
    total_loss = total_loss / len(dataloader)
    print("validation loss =", total_loss)
    return total_f1, total_loss

def predict(text, text_pipeline):
    with torch.no_grad():
        text = torch.tensor([text_pipeline(text)])
        output = model(text)
        predicted_prob = (output > 0.0).int()
        return predicted_prob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_file1 = open("positive_training_data.tsv", "r")
input_data1 = input_file1.readlines()
input_file1.close()
print("size of positive training data =", len(input_data1) - 1)

input_file2 = open("negative_training_data.tsv", "r")
input_data2 = input_file2.readlines()
input_file2.close()
print("size of negative training data =", len(input_data2) - 1)

tr_pos, tr_neg = [], []
for line in input_data1[1:]:
    words = line.strip().split('\t')
    tr_pos.append(words[2])

for line in input_data2[1:]:
    words = line.strip().split('\t')
    tr_neg.append(words[2])

combined_training_data = []

for entry in tr_pos:
    combined_training_data.append(("1", entry))

for entry in tr_neg[:10813]:
    combined_training_data.append(("0", entry))
print("size of combined training data =", len(combined_training_data))

train_iter = combined_training_data

input_file3 = open("positive_testing_data.tsv", "r")
input_data3 = input_file3.readlines()
input_file3.close()
print("size of positive testing data =", len(input_data3) - 1)

input_file4 = open("negative_testing_data.tsv", "r")
input_data4 = input_file4.readlines()
input_file4.close()
print("size of negative testing data =", len(input_data4) - 1)

combined_testing_data = []
for line in input_data3[1:]:
    words = line.strip().split('\t')
    combined_testing_data.append(("1", words[2]))

for line in input_data4[1:3036]:
    words = line.strip().split('\t')
    combined_testing_data.append(("0", words[2]))
print("size of combined testing data =", len(combined_testing_data))

test_iter = combined_testing_data

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])
print("number of dimensions =", len(vocab))

text_pipeline = lambda x: vocab(tokenize_15mer(x))
label_pipeline = lambda x: int(x)

num_class = len(set([label for (label, text) in train_iter]))
print("number of class labels =", num_class)
vocab_size = len(vocab)
emsize = 14

model = SUMO_NN_Model(vocab_size, emsize, num_class).to(device)
print("architecture of SUMO_NN model =", model)
# Hyperparameters
EPOCHS = 100  # epoch
LR = 0.05  # learning rate
BATCH_SIZE = 14  # batch size for training

class_weights = torch.tensor([1.0, 1.0])

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,50.0, gamma=0.1)
total_f1 = None

train_dataset = to_map_style_dataset(train_iter)
test_dataset = to_map_style_dataset(test_iter)
num_train = int(len(train_dataset) * 0.9)
split_train_, split_valid_ = random_split(train_dataset, [num_train, len(train_dataset) - num_train])

train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE, shuffle = True, collate_fn=collate_batch)
valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

last_loss = 100.0
trigger_times = 0
patience = 2
min_delta = 0.001

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(train_dataloader)
    f1_val, loss_val = evaluate(valid_dataloader)
    print("-" * 59)
    print(
        "| end of epoch {:3d} | time: {:5.2f}s | "
        "valid mcc {:8.3f} ".format(
            epoch, time.time() - epoch_start_time, f1_val
        )
    )
    print("-" * 59)
    if loss_val > (last_loss + min_delta):
        trigger_times += 1
        print("Trigger Times =", trigger_times)
        if trigger_times >= patience:
            print("Early stopping")
            break
    else:
        trigger_times = 0
    last_loss = loss_val
        

print("Checking the results of test dataset.")
f1_test = evaluate(test_dataloader)
print("test f1_score {:8.3f}".format(f1_test[0]))

model = model.to("cpu")

outfile = open("predictions_on_test_data.tsv", "w")
outfile.writelines("Prediction" + "\t" + "Label" + "\t" + input_data3[0])
for line in input_data3[1:]:
    words = line.strip().split('\t')
    status = predict(words[2], text_pipeline)
    outfile.writelines(str(status.item()) + "\t" + "1" + "\t" + line)

for line in input_data4[1:3036]:
    words = line.strip().split('\t')
    status = predict(words[2], text_pipeline)
    outfile.writelines(str(status.item()) + "\t" + "0" + "\t" + line)
outfile.close()
