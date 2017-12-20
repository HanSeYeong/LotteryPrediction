import cv2
import numpy as np
import csv




train_start_index = 1
train_last_index = 42
test_start_index = train_last_index
test_last_index = 53
num_in_line = 0

def calculate_num(input_x, label_y, hundred):
    if hundred == True:
        input_num = input_x[0] * 100 + input_x[1] * 10 + input_x[2]
        num_in_line = 7
    else:
        input_num = input_x[0] * 10 + input_x[1]
        num_in_line = 6
    label_num = label_y[0] * 1000 + label_y[1] * 100 + label_y[2] * 10 + label_y[3]

    return input_num, label_num

def load_reverse_sort(csvfile, start_index, end_index, hundred = True):
    count = 1
    text_tmp = []
    input_x = []
    label_y = []
    resultsForCSV = []
    data_ = csv.writer(csvfile, delimiter=' ')
    for index in range(start_index, end_index):
        text = np.loadtxt('result/' + 'result' + str(index) + '.txt')
        print(len(text))
        for j in range(len(text)):
            for i in reversed(text[j]):
                if count < 4:
                    input_x.append(int(i))
                else:
                    label_y.append(int(i))
                    if count == 7:
                        count = 1
                        input_num, label_num= calculate_num(input_x=input_x, label_y=label_y, hundred=hundred)
                        data_.writerow([input_num, label_num])
                        input_x = []
                        label_y = []
                        text_tmp = []
                        continue
                count += 1

with open('result/' + 'train_csv_','wb') as train_file:
    load_reverse_sort(csvfile=train_file, start_index=train_start_index, end_index=train_last_index, hundred=True)

# with open('result/' + 'test_csv_', 'wb') as test_file:
#     load_reverse_sort(csvfile=test_file, start_index=test_start_index, end_index=test_last_index, hundred=False)