import numpy as np



def matrix_factorization(rating, user, items, K, steps, alpha, beta):
	"""
	Input : rating matrix, user Matrix, items matrix, K number of features, number of iteration, learning rates
	Output : Recommendation matrix
	function : use an gradient descent algorithm to optimize user's value
	"""
	for step in range(steps):
		for iRow in range(np.shape(user)[0]):
			for iCol in range(np.shape(items)[1]):
				if rating[iRow][iCol] > 0:
					eij = rating[iRow][iCol] - np.dot(user[iRow,:],items[:,iCol])
					for k in range(K):
						user[iRow][k] = user[iRow][k] + alpha * (2 * eij * items[k][iCol] - beta * user[iRow][k])
						items[k][iCol] = items[k][iCol] + alpha * (2 * eij * user[iRow][k] - beta * items[k][iCol])

		e = 0
		for iRow in range(np.shape(user)[0]):
			for iCol in range(np.shape(items)[1]):
				if rating[iRow][iCol] > 0:
					e = e + pow(rating[iRow][iCol] - np.dot(user[iRow,:],items[:,iCol]), 2)
					for k in range(K):
						e = e + (beta/2) * ( pow(user[iRow][k],2) + pow(items[k][iCol],2) )
		if e < 0.001:
			break
	newR = np.dot(user, items)
	return newR

def recommendation(rating, userNumber):
	"""
	Input : rating matrix, userNumber
	Output : array list of items
	function : find the next item to recommend
	"""
	nextItem = [ 0 for i in range(len(rating[userNumber]))]
	userRate = [ 0 for i in range(len(rating[userNumber]))]
	userRate = rating[userNumber]
	#print(userRate)
	for i in range(len(rating[userNumber])):
		nextItem[i] = np.argmax(userRate) + 1
		userRate[nextItem[i] - 1] = -10000000

	return nextItem




###############################################################################

if __name__ == "__main__":
	rating = [
		[9,7,0,2],
		[4,0,0,4],
		[0,6,2,8],
		[3,0,0,1],
		[0,1,2,5],
		 ]

	rating = np.array(rating)

	k = 9
	steps=5000
	alpha=0.0002
	beta=0.02

	user = np.random.rand(len(rating),k)
	items = np.random.rand(len(rating[0]),k)
	items = np.transpose(items)
	print(rating)

	#Matrix of Recommendation
	rating = matrix_factorization(rating, user, items, k, steps, alpha, beta)
	print(rating)

	#Recommendation for users
	print("Choose a user : ")
	numb = int(input())
	print("This is the list of Recommendation : ")
	print(recommendation(rating, numb))
