import numpy as np



def matrix_factorization(Rating, User, Items, K, steps, alpha, beta):
	Items = np.transpose(Items)
	for step in range(steps):
		for iRow in range(np.shape(User)[0]):
			for iCol in range(np.shape(Items)[1]):
				if Rating[iRow][iCol] > 0:
					eij = Rating[iRow][iCol] - np.dot(User[iRow,:],Items[:,iCol])
					for k in range(K):
						User[iRow][k] = User[iRow][k] + alpha * (2 * eij * Items[k][iCol] - beta * User[iRow][k])

		e = 0
		for iRow in range(np.shape(User)[0]):
			for iCol in range(np.shape(Items)[1]):
				if Rating[iRow][iCol] > 0:
					e = e + pow(Rating[iRow][iCol] - np.dot(User[iRow,:],Items[:,iCol]), 2)
					for k in range(K):
						e = e + (beta/2) * ( pow(User[iRow][k],2) + pow(Items[k][iCol],2) )
		if e < 0.001:
			break
	newR = np.dot(User, Items)
	return newR

###############################################################################

if __name__ == "__main__":
	Rating = [
		[9,7,0,2],
		[4,0,0,4],
		[0,6,2,8],
		[3,0,0,1],
		[0,1,2,5],
		 ]

	Rating = np.array(Rating)

	K = 9
	steps=5000
	alpha=0.0002
	beta=0.02

	User = np.random.rand(len(Rating),K)
	Items = [[1,1,1,0,0,0,0,0,0],
		[1,0,0,1,0,0,0,1,0],
		[0,1,0,1,0,1,0,1,1],
		[0,0,0,1,0,1,1,1,0]
		]
	print(Rating)
	Rating = matrix_factorization(Rating, User, Items, K, steps, alpha, beta)
	print(Rating)
