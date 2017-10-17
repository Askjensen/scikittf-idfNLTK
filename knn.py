import operator
def getNeighbors(cosine_simiarities, k):
	neighbors =cosine_simiarities.argsort()[0][-k - 1:-1]
	cosine_simiarities.sort()
	distances =cosine_simiarities[0][-k - 1:-1]
	return neighbors,distances
