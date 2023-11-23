## Author : Yogendra Ramtirtha
## Email : ramtirtha.yogendra@gmail.com
## Prpject : SUMO-NN - a aneural networks based method to predict SUMOylated lysines in human proteins
## This program calculates statistical metrics to assess performance of SUMO-NN on test data

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
        count_dict[value1][value2] += 1
    return (count_dict[1][1], count_dict[1][0], count_dict[0][0], count_dict[0][1])

## This function calculates sensitivity
def sensitivity( tp, tn, fp, fn):
    return ( float(tp) / float(tp + fn) )

## This function calculates specificity
def specificity( tp, tn, fp, fn):
    return ( float(tn) / float(tn + fp) )

## This function calculates accuracy
def accuracy( tp, tn, fp, fn):
    return ( float(tp + tn) / float(tp + tn + fp + fn) )

## This program calculates F1 score
def f1_score( tp, tn, fp, fn):
    den1= float(tp + fp)
    den2= float(tp + fn)
    precision= float(tp)/den1
    recall= float(tp)/den2
    numerator= float( precision * recall)
    denominator= float(precision + recall)
    f1_score= 2 * numerator / denominator
    return f1_score

## This function calculates Matthews Correlation Coefficient
def mcc(tp, tn, fp, fn):
    numerator= ( ( float(tp) * float(tn) ) - ( float(fp) * float(fn) ) )
    denominator= ( ( ( float(tp) + float(fp) ) * ( float(tp) + float(fn) ) * ( float(tn) + float(fp) ) * ( float(tn) + float(fn) ) ) ** 0.5 )
    mcc= numerator / denominator
    return mcc

input_file = open("predictions_on_test_data.tsv", "r")
input_data = input_file.readlines()
input_file.close()
print("size of predictions file =", len(input_data) - 1)

predictions, labels = [], []
tp, fp, tn, fn = 0, 0, 0, 0

for line in input_data[1:]:
    words = line.strip().split('\t')
    predictions.append(int(words[0]))
    labels.append(int(words[1]))

print("\n")

tp, fp, tn, fn = get_confusion_matrix(predictions, labels)

print("tp, fp, tn, fn =", tp, fp, tn, fn)

print("\n")

print("sensitivity =", sensitivity(tp, tn, fp, fn))

print("specificity =", specificity(tp, tn, fp, fn))

print("accuracy =", accuracy(tp, tn, fp, fn))

print("f1_score =", f1_score(tp, tn, fp, fn))

print("mcc =", mcc(tp, tn, fp, fn))
