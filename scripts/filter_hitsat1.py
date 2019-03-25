all_hits1_file = 'test_all_hits1_id.txt'
x_file = 'test_x_mapped.txt'
y_file = 'test_y_true.txt'
y1_file = 'test_y_true_single_label.txt'

all_hits1 = set(map(str.strip,open(all_hits1_file).readlines()))
x = map(str.strip,open(x_file).readlines())
y = map(str.strip, open(y_file).readlines())
y1 = map(str.strip,open(y1_file).readlines())
fx = []
fy = []
fy1 = []
for tx,ty,ty1 in zip(x,y,y1):
    if tx in all_hits1:
        fx.append(tx)
        fy.append(ty)
        fy1.append(ty1)
    else:
        print("skipping...not hits at 1")

fhx = open('test_x_mapped_hits1.txt','w')
fhy = open('test_y_true_hits1.txt','w')
fhy1 = open('test_y_true_single_label_hits1.txt','w')

print('\n'.join(fy),file=fhy)
print('\n'.join(fy1),file=fhy1)
print('\n'.join(fx),file=fhx)
fhy.close()
fhx.close()
fhy1.close()

