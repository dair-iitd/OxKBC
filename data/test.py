def spiral_scan(A):
	if(len(A)==0):
		return []
	elif(len(A)==1):
		return A[0]
	elif(len(A)==2):
		tmp1 = A[0]
		tmp2 = A[1].copy()
		tmp2.reverse()
		return tmp1+tmp2
	elif(len(A[0])==1):
		tmp = []
		for j in range(len(A)):
			tmp.append(A[j][0])
		return tmp
	elif(len(A[0])==2):
		answer = A[0].copy()
		for j in range(1, len(A)-1):
			answer.append(A[j][1])
		tmp = A[len(A)-1].copy()
		tmp.reverse()
		answer.extend(tmp)
		for j in range(len(A)-2, 0, -1):
			answer.append(A[j][0])
		return answer
	answer = A[0].copy()
	for j in range(1, len(A)-1):
		answer.append(A[j][len(A[0])-1])
	tmp = A[len(A)-1].copy()
	tmp.reverse()
	answer.extend(tmp)
	for j in range(len(A)-2, 0, -1):
		answer.append(A[j][0])
	newA = []
	for i in range(1, len(A)-1):
		newA.append([])
		for j in range(1, len(A[0])-1):
			newA[-1].append(A[i][j])
	answer.extend(spiral_scan(newA))
	return answer
A = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
print(spiral_scan(A))

