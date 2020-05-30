import numpy
from scipy.io import loadmat, savemat
import matplotlib.pyplot as pl
import cv2
'''
t1 = loadmat("Theta1.mat")["Theta1"]  # 25*401
t2 = loadmat("Theta2.mat")["Theta2"]  # 10*26
'''

def frontPropagation(imagen,t1,t2):
    m, n = imagen.shape #5300x400
    a1 = imagen  # capa 1
    aux = numpy.ones((n+1, m))
    aux[1:, :] = imagen.T #401x5300
    a1 = aux
    h1=a1
    a2 = t1.dot(a1)  # 25*5300
    a2 = sigmoide(a2)
    aux = numpy.ones((26, m)) #26x5300
    aux[1:, :] = a2
    a2 = aux
    h2=a2
    a3 = t2.dot(a2)  # 10*5300 prediccion
    a3 = sigmoide(a3)
    h3 = a3
    return (h1,h2,h3)


def sigmoide(h):
    return 1. / (1 + numpy.e ** (-h))

def convYs(y):
        resp=[]
        for i in y:
            lista = [(lambda x: lambda y: 1 if x==y else 0)(x)(i) for x in range(1,11)]
            resp+=[lista]
        return numpy.array(resp)

def funcionCosto(X, y, t1, t2, paramLam):
    m, n = X.shape 
    h = frontPropagation(X,t1, t2)[-1]
    y_vec = numpy.zeros((m,12))
    for i in range(1,13):
        y_vec[:, i-1] = (y == i).flatten()
    j = (- 1. / m * (y_vec * numpy.log(h.T) + (1-y_vec)* numpy.log(1 - h.T))).sum() + paramLam
    return(j)
def funcionCosto2(X,y,t1,t2):
    h = frontPropagation(X,t1, t2)[-1]
    y_vec = numpy.zeros((m,10))
    for i in range(1,11):
        y_vec[:, i-1] = (y == i).flatten()
    j = numpy.mean((h-y_vec.T)**2)
    return j

#gradiente necesita de back propagation
def gradiente(X,y, t1, t2, paramLam):
    m, n = X.shape
    #computar h
    h1,h2,h3 = frontPropagation(X, t1, t2)
    #computar delta (error) ultima capa
    y_vec = numpy.zeros((m,12))
    for i in range(1,13):
        y_vec[:, i-1] = (y == i).flatten()
    
    delta3 = (h3 - y_vec.T)#10x5000
    delta2 = t2.T.dot(delta3) *(h2*(1-h2)) #26x5000
    aux_delta2 = delta2[1:,:]

    t1 = t1 - paramLam /m * (aux_delta2).dot(h1.T)
    t2 = t2 - paramLam /m * (delta3).dot(h2.T)
    return t1, t2
    
def entrenamiento(X, y, t1,t2, ls, epsilon=10**(-6)):
    historial = [funcionCosto(X,y,t1,t2,ls)]
    iteraciones = 1
    while(True):
        #print(historial[-1])
        t1, t2 = gradiente(X, y, t1, t2, ls)
        historial.insert(len(historial),funcionCosto(X,y,t1,t2,ls))
        #if(abs(historial[-1]-historial[-2]) < epsilon):
            #break
        if(iteraciones > 2000):
            break
        iteraciones+=1
    return [t1,t2],historial


data = loadmat("Entradas.mat")
X = data["X"]
y = data["y"]
m, n = X.shape
#topollogia red
capa_entrada = n+1
capa_salida = 12
capa_oculta = 26
#thetas
Theta = numpy.random.rand(capa_entrada * (capa_oculta-1) + capa_oculta*capa_salida)*2-1
t1 = Theta[0:(25*401)].reshape(25,401)
t2 = Theta[(25*401):].reshape(12, 26)
#entrenamiento
thetas, hist = entrenamiento(X,y,t1,t2,1)
#grafica descenso de gradiente
xs = numpy.linspace(0,100,len(hist))
pl.plot(xs,hist)
pl.show()
#prediccion entradas X
x = X[2560].reshape(1,400)
pred = frontPropagation(x,thetas[0],thetas[1])[-1].flatten()
indMax = pred.argmax()
print (indMax+1, " con certeza: ",pred[indMax]*100)
print(y[2560])

#prediccion entradas X
x = X[3560].reshape(1,400)
pred = frontPropagation(x,thetas[0],thetas[1])[-1].flatten()
indMax = pred.argmax()
print (indMax+1, " con certeza: ",pred[indMax]*100)
print(y[3560])

## Datos entrenamiento
print("Dos ultimos Js: ",hist[-1],hist[-2],"\ncon iteraciones: ",len(hist))


# Guardar Prametros
aux = dict()
aux["t1"] = thetas[0]
aux["t2"] = thetas[1]
savemat("thetas.mat", aux)






