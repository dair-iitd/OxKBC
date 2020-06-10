f1 = open("../../../prachi/kbi/kbi-pytorch/hits_id/fb15k_hits_10_pred.txt",'r').readlines()
f2 = open("../../../prachi/kbi/kbi-pytorch/hits_id/fb15k_hits_10_true.txt",'r').readlines()

f3 = open("fb15k_hits10_not_hits1_pred.txt",'w')
for i in range(len(f1)):
    if(f1[i]!=f2[i]):
        print(f1[i].strip(),file=f3)
f3.close()
