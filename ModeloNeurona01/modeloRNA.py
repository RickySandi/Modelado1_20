import numpy
from scipy.io import loadmat
import cv2
from scipy.optimize import minimize

t1 = loadmat("Theta1.mat")["Theta1"]  # 25*401
t2 = loadmat("Theta2.mat")["Theta2"]  # 10*26

def sigmoide(h):
    return 1. / (1 + numpy.e ** (-h))


def frontPropagation(imagen,t1,t2):

    m, n = imagen.shape
    a1 = imagen  # capa 1
    #aniadir 1
    aux = numpy.ones((n+1, m))
    aux[1:, :] = imagen.T
    a1 = aux

    a2 = t1.dot(a1)  # 25*1
    a2 = sigmoide(a2)
    #aniadir 1
    aux = numpy.ones((26, 1))
    aux[1:, :] = a2
    a2 = aux

    a3 = t2.dot(a2)  # 10*1 prediccion
    a3 = sigmoide(a3)
    h = a3
    return (h)

def randInitializeWeights(L_in, L_out, epsilon_init= 0.12):

    W = numpy.zeros((L_out, 1+L_in))
    W = numpy.random.rand(L_out, 1+L_in) * 2 * epsilon_init - epsilon_init
    return W

def sigmoidGradient(z):

    g = sigmoide(z) * (1 - sigmoide(z))
    return g 

lam_parm = 0.01
def funcionCosto(X, y, theta):
    m, n = X.shape 

    t1 = theta[0:(25 * 401)].reshape(25, 401)
    t2 = theta[(25 * 401):].reshape(10, 26)
  
   # print(t1.shape, " ", t2.sahpe)
    h = frontPropagation(X,t1, t2) #10*5000
    y_vec = numpy.zeros((m, 10))
    for i in range(1, 11):
        y_vec[:, i-1] = (y == i).flatten() #y_vec = 5000 * 10

    J = (-1. /m * (y_vec * numpy.log(h.T) + (1-y_vec) * numpy.log(1-h.T)))
    #param_reg = lam_parm / (2*m)
    return J 

    #para computar el gradiente implementaremos backPropagation 

def gradiente(X,y,theta):
        m, n = X.shape
        t1 = theta[0:(401 * 25)].reshape(25,401)
        t2 = theta[(401 * 26):].reshape(10,26)

        y_vec = numpy.ones((m,10))
        for i in range(1,11):
            y_vec[:, i-1] = (y == i).flatten() #5000 * 10

        ## frontPropagation
        a1 = X #capa 1
        aux = numpy.ones((n+1, m))
        aux[1:, :] = X.T

        a1 = aux #401*5000
        #print (a1.shape)
        # t1 (25*401)  a1(401*5000) -> a2 (25*5000) 
        a2 = t1.dot(a1)  # 25*m
        a2 = sigmoide(a2)

        aux = numpy.ones((26, m))
        aux[1:, :] = a2
        a2 = aux

        a3 = t2.dot(a2)  # 10*26   prediccion
        a3 = sigmoide(a3)
        h = a3 #10*5000
        ##################
        #computar error en la ultima capa
        delta3 = h - y_vec.T    #delta3 = 10*5000

        #computar errores anterior capa
        delta2 = t2.T.dot(delta3) * sigmoidGradient(a2) #25*5000

        Delta1 = delta2[1:,:].dot(a1.T)     #25*5000  401*5000 -> 25*401
        Delta2 = delta3.dot(a2.T)           #10*5000  26*5000 -> 10*26

        THETA_grad_1 = 1. / m * (Delta1)
        THETA_grad_2 = 1. / m * (Delta2)
    
        return numpy.concatenate([THETA_grad_1.flatten(), THETA_grad_2.flatten()]) #10285






    
   

    
data = loadmat("digitos.mat")
X = data["X"]
y = data["y"]
m,n = X.shape
# topologia dela red
capa_entrada = n+1
capa_salida = 10
capa_oculta = 26
##################### 
Theta = numpy.ones(capa_entrada * (capa_oculta-1) + capa_oculta * capa_salida)
print (Theta.shape)
print (funcionCosto(X ,y, Theta))