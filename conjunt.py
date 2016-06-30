from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn import svm
from sklearn.svm import SVC
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from pyevolve import Util
from pyevolve import GTree
from pyevolve import GSimpleGA
from pyevolve import Consts
import numpy as np
import operator

global_vector= []
data_set = "pimas.txt"
#data_set = "australian.txt"
#data_set = "german.txt"
#data_set = "wdbc.txt"
#data_set = "iono.txt"
best_val = 1
valores = []
fitness = []

def evaluation(chromosome):
	global best_val
	global valores
	global a
	global b
	global c
	global d
	global e
	global f
	code_comp = chromosome.getCompiledCode()
	features = eval(code_comp)
	cfeatures = len(features)
	matrix_final = reducir(matrix,features)
	evaluated_data = svm(matrix_final)
	total = len(matrix[0])-1
	alfa = 0.5
	beta = 0.4
	gama = 0.1
	valor = ((alfa*(1-evaluated_data[0]))+beta*(1-evaluated_data[1])+gama*(cfeatures/total))/3.0	
   	if(valor < best_val):
		best_val = valor
		valores = []
		valores.append(evaluated_data[0])
		valores.append(evaluated_data[1])
		valores.append(cfeatures)
	fitness.append(valor)
	print 'AUC: ',evaluated_data[0],'ACC: ',evaluated_data[1]
	return valor
	
def reducir(matrix,features):
	matrix_reducida = []
	for row in matrix:
		row_final = []
		for feature in features:
			row_final.append(row[feature])
		row_final.append(row[len(row)-1])
		matrix_reducida.append(row_final)
	return matrix_reducida

def svm(matrix):
   # se transforma la ultima columna a un array de float
   last_column = [row[len(matrix[0])-1] for row in matrix]
   data_class = transform_to_int(last_column, matrix[0][len(matrix[0])-1])
   # se transforma los datos de una lista a un array float
   indices = list(range(len(matrix[0])-1))
   new_list = map(operator.itemgetter(*indices), matrix)
   data = np.asarray(new_list) 
   data = data.astype(np.float)
   # se realiza cross-validation y svm
   data_train, data_test, class_train, class_test = cross_validation.train_test_split(data, data_class, test_size=0.5, random_state=0)
   if data_train.ndim == 1:
      data_train = np.reshape(data_train, (len(data_train),1 ))
      data_test = np.reshape(data_test, (len(data_test),1 ))
   data_train = np.array(data_train)
   data_train = preprocessing.StandardScaler().fit(data_train).transform(data_train)
   data_test = np.array(data_test)
   data_test = preprocessing.StandardScaler().fit(data_test).transform(data_test)
   clf = SVC(kernel='rbf',gamma=1 ,C=1,probability=True,shrinking=False).fit(data_train, class_train)
   predicted = clf.predict(data_test)
   # se calcula ACC
   ACC = accuracy_score(class_test, predicted)
   score = clf.score(data_test,class_test)
   # se calcula AUC
   fpr, tpr, thresholds = roc_curve(class_test, predicted)
   area = auc(fpr, tpr)
   results = []
   results.append(area)
   results.append(ACC)
   return results

#GP functions
def gp_intersect(a,b):
	lista = [val for val in a if val in b]
	if lista:
		return lista
	else:
		if len(a) <= len(b):
			return a
		else:
			return b
		
def gp_diference(a,b):
	lista = [val for val in a if val not in b]
	if lista:
		return lista
	else:
		if len(a) <= len(b):
			return a
		else:
			return b

def gp_union(a,b):
	lista = []
	for data in a:
		lista.append(data)
	for data in b:
		if data not in lista:
			lista.append(data)
	return lista

def gp_simetric(a,b):
	intersection = [val for val in a if val in b]
	lista = []
	for data in a:
		if data not in intersection:
			lista.append(data)
	for data in b:
		if data not in intersection:
			lista.append(data)
	
	if lista:
		return lista
	else:
		if len(a) <= len(b):
			return a
		else:
			return b

def gp_complement(a):
	lista = [val for val in global_vector if val not in a]
	if lista:
		return lista	
	else:
		return a

#Se transforma la columna de string a enteros
def transform_to_int(lista, letra):
   result = []
   for value in lista:
      if value == letra:
         result.append(1)
      else:
         result.append(0)
   result = np.asarray(result)
   result = result.astype(np.float)
   return result

# Para leer el dataset
def readAndParse(fileName):
   file = open(fileName,'r')
   parsedLine = []
   parsedLines = []
   for line in file:
      parsedLine = line.split(' ')
      i = 0
      for data in parsedLine:
         parsedLine[i] = data.strip()
         i = i + 1
      parsedLines.append(parsedLine)
   return parsedLines

#GP Functions
# Feature selection basado en la varianza de cada caracteristica
def feat1(matrix):
	last_column = [row[len(matrix[0])-1] for row in matrix]
	data_class = transform_to_int(last_column, matrix[0][len(matrix[0])-1])
	indices = list(range(len(matrix[0])-1))
	new_list = map(operator.itemgetter(*indices), matrix)
	data = np.asarray(new_list) 
	data = data.astype(np.float)
	sel = VarianceThreshold(threshold=(0.35))
	matrix_new =  sel.fit_transform(data)
	data_class = np.array([data_class])
	features_selected = np.concatenate((matrix_new,data_class.T),axis=1)
	indices_resultados = sel.get_support(new_list)
	features = []	
	for data in indices_resultados:
		features.append(data)
	return features

# Feature selection basado en un test estadistico univariado por feature utilizando k vecinos mas cercanos

def feat2(matrix):
	last_column = [row[len(matrix[0])-1] for row in matrix]
	data_class = transform_to_int(last_column, matrix[0][len(matrix[0])-1])
	indices = list(range(len(matrix[0])-1))
	new_list = map(operator.itemgetter(*indices), matrix)
	data = np.asarray(new_list) 
	data = data.astype(np.float)
	features = len(data[0])-1
	matrix_new = SelectKBest(chi2, k=4).fit_transform(data, data_class)
	indices_resultados =  SelectKBest(chi2, k=4).fit(data, data_class).get_support(new_list)
	data_class = np.array([data_class])
	features_selected = np.concatenate((matrix_new,data_class.T),axis=1)
	features = []	
	for data in indices_resultados:
		features.append(data)
	return features

# Feature selection recursive feature elimination
def feat3(matrix):
	last_column = [row[len(matrix[0])-1] for row in matrix]
	data_class = transform_to_int(last_column, matrix[0][len(matrix[0])-1])
	indices = list(range(len(matrix[0])-1))
	new_list = map(operator.itemgetter(*indices), matrix)
	data = np.asarray(new_list) 
	data = data.astype(np.float)
	svc = SVC(kernel="linear", C=1)
	rfe = RFE(estimator=svc, n_features_to_select=5, step=1)
	matrix_new = rfe.fit_transform(data, data_class)
	data_class = np.array([data_class])
	features_selected = np.concatenate((matrix_new,data_class.T),axis=1)
	indices_resultados = rfe.get_support(new_list) 
	features = []	
	for data in indices_resultados:
		features.append(data)
	return features

def classbalance(matrix):
   data_class = []
   for i in xrange(0,len(matrix)):
      data_class.append(matrix[i][len(matrix[0])-1])

   data_class = np.asarray(data_class)
   data_class = data_class.astype(np.float)
   data = np.delete(matrix,(len(matrix[0])-1), axis=1)
# se cuenta la cantidad de elementos en cada clase
   positive = 0
   negative = 0
   for value in data_class:
      if value == 1:
         positive += 1
      else:
         negative += 1
   data_class = np.array([data_class])
   data_array = np.concatenate((data,data_class.T),axis=1)
   data_sort = data_array[np.argsort(data_array[:,len(data_array[0])-1])]
   data_sort = np.array(data_sort).tolist()
   if positive == negative:
      data_sort = np.asarray(data_sort)
      data_sort = data_sort.astype(np.float)
      return data_sort
 # eliminar filas del inicio
   if positive < negative:
      borrar = negative - positive
      i = 0
      while i < borrar:
         del(data_sort[0])
         i += 1
      data_sort = np.asarray(data_sort)
      data_sort = data_sort.astype(np.float)
      return data_sort
 # eliminar filas del final   
   else:
      borrar = positive - negative
      empezar = len(data_sort) - borrar
      len_original = len(data_sort)
      aux = empezar
      while empezar < len_original:
         del(data_sort[aux])
         empezar += 1
      data_sort = np.asarray(data_sort)
      data_sort = data_sort.astype(np.float)
      return data_sort

matrix = readAndParse(data_set)
a = feat1(matrix)
b = feat2(matrix)
c = feat3(matrix)
d = feat1(classbalance(matrix))
e = feat2(classbalance(matrix))
f = feat3(classbalance(matrix))

def main_run():
	genome = GTree.GTreeGP()
	genome.setParams(max_depth=5, method="grow") 
	genome.evaluator.set(evaluation)
	ga = GSimpleGA.GSimpleGA(genome)
	ga.setParams(gp_terminals = ['a','b','c','d','e','f'], gp_function_prefix = "gp")
	ga.setMinimax(Consts.minimaxType["minimize"])
	ga.setGenerations(50)
	ga.setCrossoverRate(1.0)
	ga.setMutationRate(0.25)
	ga.setPopulationSize(50)
	ga(freq_stats=1)
	best = ga.bestIndividual()
	best.writeDotImage("trees.jpg")
	print best

if __name__ == "__main__":
	matriz = readAndParse(data_set)
	for i in xrange(0,len(matriz[0])-1):
		global_vector.append(i)
	main_run()
	print valores
	chunk = [fitness[x:x+3] for x in xrange(0, len(fitness), 3)]
	salida = open('salida.csv','w')
	salida.write(',max,min,average'+'\n')
	i=1
	for fit in chunk:
		salida.write(str(i)+','+str(max(fit))+','+str(min(fit))+','+str(float(sum(fit))/float(len(fit)))+'\n')
		i += 1
	salida.close()
