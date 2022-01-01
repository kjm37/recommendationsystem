import numpy as np
import pandas as pd
from web.models import Myrating
import scipy.optimize


def Myrecommend():
	def normalizeRatings(myY, myR):
    	# The mean is only counting movies that were rated
		Ymean = np.sum(myY,axis=1)/np.sum(myR,axis=1)
		Ymean = Ymean.reshape((Ymean.shape[0],1))
		return myY-Ymean, Ymean
	
	#this function flat all array and combine in one array
	def flattenParams(myX, myTheta):
		return np.concatenate((myX.flatten(),myTheta.flatten()))
    
	#this functions retirn predicted values and convert it into seprate and flatten
	def reshapeParams(flattened_XandTheta, mynm, mynu, mynf):
		assert flattened_XandTheta.shape[0] == int(mynm*mynf+mynu*mynf)	#assert check condition is true or not and here we convert all passed data to integer and create array size like number of movies into 10 and number of users into 10
		reX = flattened_XandTheta[:int(mynm*mynf)].reshape((mynm,mynf))#reshape our output like number of movies *10 (here 10 is we want to show 10result so)
		reTheta = flattened_XandTheta[int(mynm*mynf):].reshape((mynu,mynf))#reshape our output like number of movies *10 (here 10 is we want to show 10result so)
		return reX, reTheta

	def cofiCostFunc(myparams, myY, myR, mynu, mynm, mynf, mylambda = 0.):
		myX, myTheta = reshapeParams(myparams, mynm, mynu, mynf)
		term1 = myX.dot(myTheta.T)
		term1 = np.multiply(term1,myR)
		cost = 0.5 * np.sum( np.square(term1-myY) )
    	# for regularization
		cost += (mylambda/2.) * np.sum(np.square(myTheta))
		cost += (mylambda/2.) * np.sum(np.square(myX))
		return cost



	def cofiGrad(myparams, myY, myR, mynu, mynm, mynf, mylambda = 0.):
		myX, myTheta = reshapeParams(myparams, mynm, mynu, mynf)
		term1 = myX.dot(myTheta.T)	#return the dot product
		term1 = np.multiply(term1,myR)
		term1 -= myY
		Xgrad = term1.dot(myTheta)	#return dot product
		Thetagrad = term1.T.dot(myX)
    	# Adding Regularization
		Xgrad += mylambda * myX
		Thetagrad += mylambda * myTheta
		return flattenParams(Xgrad, Thetagrad)



	# programs execution starts here
	df=pd.DataFrame(list(Myrating.objects.all().values())) #create dataframe of myrating table values

	print("df of rating :" ,df)
	mynu=df.user_id.unique().shape[0]		#get all users id
	print("mynu",mynu)
	mynm=df.movie_id.unique().shape[0]		#get all movie id
	print("mynm", mynm)
	mynf=10									# for top 10 
	Y=np.zeros((mynm,mynu))					#create array of dimension[no_of_user][number_of_movies]
	for row in df.itertuples():
		Y[row[2]-1, row[4]-1] = row[3]		#adding rating values in Y array
	R=np.zeros((mynm,mynu))					#create zero array with dimension of same as Y
	for i in range(Y.shape[0]):				#iterate all rows of Y array
		for j in range(Y.shape[1]):			#iterate colu  ms of y aray
			if Y[i][j]!=0:
				R[i][j]=1






	Ynorm, Ymean = normalizeRatings(Y,R)		#call function and pass Y and R arrayand return values
	X = np.random.rand(mynm,mynf)				#shuffle all values
	Theta = np.random.rand(mynu,mynf)
	myflat = flattenParams(X, Theta)			#call function and passX and theta
	mylambda = 12.2
	result = scipy.optimize.fmin_cg(cofiCostFunc,x0=myflat,fprime=cofiGrad,args=(Y,R,mynu,mynm,mynf,mylambda),maxiter=40,disp=True,full_output=True) #inbuilt model
	resX, resTheta = reshapeParams(result[0], mynm, mynu, mynf)	#call function with results and numberofmovies,numberofusers and top items
	prediction_matrix = resX.dot(resTheta.T)	#return prediction matrix of result
	print("prediction matrix :" ,prediction_matrix)
	

	###################################################

	pred_values = resX.flatten()			#store predicted values in variable
	print("RESULT X :",pred_values)
	print("LENGTH OF RESUT :",len(pred_values))



	train = myflat[:len(pred_values)]		#flatten the array of predicted size
	print("myflat:",train)
	print("length of myflat : ",len(train))
	
	


	y_train = [int(i) for i in train]		#convert array into int array
	y_test = [int(i) for i in pred_values]	#convert array into int array

	accuracy = sum(1 for x,y in zip(y_train,y_test) if x == y) / float(len(y_train)) 	#find accuracy using total predicted and training match devided by total traininng data
	
	print("accuracy : ", accuracy)

	#######################################################
	
	return prediction_matrix,Ymean			#return prediction matrix


###############################################################################
#how we find  ACCURACY
	# for finding accuracy we need training and testing data so here we use our movie ratings from our database is a traing data
	# and after prediction we got predicted data so we use this data against training Data
	# that means we have original output and predicted output so we compare both original and predicted output as follows-
	# we add all the match of original and predicted values if match then we increase else pass
	# we match like this - if in out training we have same data in prediction then we increase
	# here we use  :
	# 	zip(y_train,y_test) - this iterate two lists training outputs and predicted outputs
	# 	1 for x,y in zip(y_train,y_test) - here we store corresponding values in train and predicted values in x and y
	# 	sum(1 for x,y in zip(y_train,y_test) if x == y) - here we use sum function to make sum of the matches and we use if for match check if match then increase
	# 	 float(len(y_train)) - this find the length of total size of training  Data

	# 	sum(1 for x,y in zip(y_train,y_test) if x == y) / float(len(y_train)) - here we find accuracy between original daata value and predicted data Value like we find overall percentage








