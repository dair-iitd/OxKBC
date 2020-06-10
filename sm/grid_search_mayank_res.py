import os
datasets = ["yago"]
supervisions = ["sup","un","semi"]
num_runs = 3
folders_to_analyze = ["test","test_ml"]

# print("setting - test_mean - test_stddev - test_ml_mean - test_ml_stddev")
print("setting - test_mean - test_stddev")

os.chdir("grid_search")
for dataset in datasets:
	os.chdir(dataset)
	for supervision in supervisions:
		os.chdir(supervision)		
		all_settings = os.listdir()
		print(dataset,supervision)
		for setting in all_settings:
			os.chdir(setting)
			test_scores = []
			test_ml_scores = []
			for run in range(1,num_runs+1):
				os.chdir("run_"+str(run))
				for folder in folders_to_analyze:
					os.chdir(folder)	
					f = open("log.txt",'r')
					score = float(f.readlines()[-2].split(',')[-1].strip())
					f.close()
					if folder=="test":
						test_scores.append(score)
					else:	
						test_ml_scores.append(score)	
					os.chdir("..")
				os.chdir("..")
			os.chdir("..")
			test_mean = sum(test_scores)/num_runs	
			test_ml_mean = sum(test_ml_scores)/num_runs
			test_stddev = 0
			test_ml_stddev = 0
			for i in range(num_runs):
				test_stddev += test_scores[i]*test_scores[i]
				test_ml_stddev += test_ml_scores[i]*test_ml_scores[i]
			test_stddev	/= num_runs	
			test_ml_stddev	/= num_runs	

			test_stddev	-= test_mean**2
			test_ml_stddev	-= test_ml_mean**2

			if(abs(test_stddev)<0.0000001):
				test_stddev=0
			else:
				test_stddev	= test_stddev**0.5
			test_ml_stddev	= test_ml_stddev**0.5
			# print(setting,"\t",round(test_mean,3),"\t",round(test_stddev,3),"\t",round(test_ml_mean,3),"\t",round(test_ml_stddev,3))
			print(setting,"\t\t",round(test_mean,3),"\t",round(test_stddev,3))
		os.chdir("..")
	os.chdir("..")






