import argparse
import random


#get n random points out of lines
def get(lines, n):
	assert(n<=len(lines))
	random.shuffle(lines)
	return lines[:n]



if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-f1', '--hits10', required=True, default=None)
	# parser.add_argument('-f2', '--not_hits10', required=True, default=None)
	parser.add_argument('-op', '--output_path', required=True, default=None)	
	args = parser.parse_args()

	f1 = open(args.hits10,'r').readlines()
	# f2 = open(args.not_hits10,'r').readlines()

	#total 450, such that hits1 ~ 300
	random.shuffle(f1)
	a1 = get(f1, 170)
	# a2 = get(f2, 53)
	# answer = a1 + a2
	answer = a1

	f3 = open(args.output_path,'w')
	for i in answer:
		print(i.strip(),file=f3)
	f3.close()


