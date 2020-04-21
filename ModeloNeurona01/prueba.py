import numpy
from scipy.io import loadmat
import cv2

t1 = loadmat("Theta1.mat")["Theta1"]  # 25*401
t2 = loadmat("Theta2.mat")["Theta2"]  # 10*26


def frontPropagation(imagen,t1,t2):
    m, n = imagen.shape
    a1 = imagen  # capa 1
    aux = numpy.ones((n+1, m))
    aux[1:, :] = imagen.T
    a1 = aux
    a2 = t1.dot(a1)  # 25*1
    a2 = sigmoide(a2)

    aux = numpy.ones((26, 1))
    aux[1:, :] = a2
    a2 = aux

    a3 = t2.dot(a2)  # 10*1 prediccion
    a3 = sigmoide(a3)
    h = a3
    return (h)


def sigmoide(h):
    return 1. / (1 + numpy.e ** (-h))

def funcionCosto(X, y, theta):
    m, n = X.shape 
    t2 = theta[(25 * 401):].reshape(10, 26)
    t1 = theta[0:(25 * 401)].reshape(25, 401)
   # print(t1.shape, " ", t2.sahpe)
    h = frontPropagation(X,t1, t2) #10*5000
    y_vec = numpy.zeros((m, 10))
    for i in range(1, 11):
        y_vec[:, i-1] = (y == i).flatten()

    J = (-1. /m * (y_vec * numpy.log(h.T) + (1-y_vec) * numpy.log(1-h.T))).sum()
    return J 
    
   

    
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












'''
imagen = cv2.imread("test07.jpg")
imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
imagen = imagen.T.flatten()  # vector 400 posiciones
imagen = imagen.reshape(imagen.size, 1)
prediccion, probabilidad = frontPropagation(imagen)

print(prediccion, " con ", probabilidad)
'''

'''
imagen = cv2.imread("prueba3.jpg")
imagenGris = cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY)
imagenGris = cv2.GaussianBlur(imagenGris, (5,5), 0)

ret, imagenBN = cv2.threshold(imagenGris, 90, 225, cv2.THRESH_BINARY_INV)

grupos, i = cv2.findContours(imagenBN.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
ventanas = [cv2.boundingRect(g) for g in grupos]

for g in ventanas:
    cv2.rectangle(imagen, (g[0], g[1]), (g[0]+g[2], g[1]+g[3]),(255,0,0),2)
    l = int(g[3] * 1.6)
    p1 = int(g[1] + g[3] // 2) - l // 2
    p2 = int(g[0] + g[2] // 2) - l // 2
    digito = imagenBN[p1: p1+l, p2: p2+l]
    digito = cv2.resize(digito, (20,20), interpolation = cv2.INTER_AREA)
    digito = cv2.dilate(digito, (3,3,))
    aux = digito.T.flatten()
    aux = aux.reshape(aux.size,1)
    prediccion, probabilidad = frontPropagation(aux)
    print(prediccion, " con ", probabilidad)
    cv2.imshow("d", digito)
    cv2.waitKey()
cv2.imshow("Digitos", imagen)
cv2.waitKey() 
'''