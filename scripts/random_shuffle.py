import pickle
import sys
dir=sys.argv[1]
input_file_path = dir+'sm_valid_with_id.pkl'
data = pickle.load(open(input_file_path, 'rb'))
import numpy as np
np.random.shuffle(data)
sup_train = data[:50]
sup_valid = data[50:]
pickle.dump(sup_train,open(dir+'sm_sup_train_with_id.pkl','wb'))
pickle.dump(sup_valid,open(dir+'sm_sup_valid_with_id.pkl','wb'))
