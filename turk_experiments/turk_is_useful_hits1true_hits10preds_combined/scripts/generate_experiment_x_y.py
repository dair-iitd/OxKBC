"""
This script takes 150 elements of hits1_true(TRUE) and 150 elements of hits10_nothits1(FALSE)
Shuffles them
And writes the X and Y file with 300 elements each
"""
import random

f1 = open("../fb15k_hits_1_true_named.txt_id.txt",'r').readlines()
f2 = open("../fb15k_hits10_not_hits1_pred_named.txt_id.txt",'r').readlines()

f3 = open("../is_useful_x.txt",'w')
f4 = open("../is_useful_y.txt",'w')

random.shuffle(f1)
random.shuffle(f2)

f1 = f1[:150]
f2 = f2[:150]

new_point = []
for f in f1:
    new_point.append([f,1])
for f in f2:
    new_point.append([f,0])

random.shuffle(new_point)
for i in range(len(new_point)):
    print(new_point[i][0].strip(),file=f3)
    print(new_point[i][1],file=f4)
f3.close()
f4.close()
