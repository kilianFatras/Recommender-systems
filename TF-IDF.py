import numpy as np



def tf_idf(user, itemSelected, item, numberAliment):
	"""input : user matrix, item selected matrix, item matrix
	   output : user matrix
	   function : determine the weigth of each ingredient
	"""
	for idAliment in range(numberAliment):
		freq = 0
		for idItem in itemSelected:
			if idItem[idAliment] == 1:
				freq += 1
		user[idAliment] = float(freq/len(itemSelected))

	for idAliment in range(numberAliment):
		freq = 0
		for idItem in item:
			if idItem[idAliment] == 1:
				freq += 1
		user[idAliment] = user[idAliment] * np.log(len(item)/freq)


	return user


def recommandation(user, item):
	"""input : user matrix, item matrix
	   output : recommandation of items
	   function : find the items to recommend
	"""
	score = []
	recommandation = []
	for idItem in item:
		score.append(np.dot(user, idItem))
	for i in range(len(score)):
		recommandation.append(np.argmax(score))
		score[recommandation[i]] = 0

	return recommandation



#####################################################
if __name__ == '__main__':

#####Data set
	item = np.random.randint(2, size=(10, 4)) #10 plats avec 4 aliments différents (on peut retrouver les mêmes plats)
	user = np.random.rand(4)
	itemSelected = []
	recommend = []
	wish = 0
	print("select your item, enter 1 if you want the item, else 0 : ")
	for i in range(10):
		print(item[i])
		wish = int(input())
		if wish == 1 :
			itemSelected.append(item[i])
	user = tf_idf(user, itemSelected, item, 4)
	recommend = recommandation(user, item)
	print(recommend)
