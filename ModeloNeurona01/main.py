import numpy
from scipy.io import loadmat
import cv2

t1 = loadmat("Theta1.mat")["Theta1"]  # 25*401
t2 = loadmat("Theta2.mat")["Theta2"]  # 10*26


def frontPropagation(imagen=""):
    a1 = imagen  # capa 1
    aux = numpy.ones((401,1))
    aux[1:, :] = imagen
    a1 = aux

    print(a1.shape)

    a2 = t1.dot(a1)  # 25*1
    a2 = sigmoide(a2)

    aux = numpy.ones((26, 1))
    aux[1:, :] = a2
    a2 = aux

    a3 = t2.dot(a2)  # 10*1 prediccion
    a3 = sigmoide(a3)
    digi = a3.argmax() + 1
    proba = a3[digi - 1] * 100
    return digi, proba


def sigmoide(h):
    return 1. / (1 + numpy.e ** (-h))


imagen = cv2.imread("test23.jpg")
imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
imagen = imagen.T.flatten()  # vector 400 posiciones
imagen = imagen.reshape(imagen.size, 1)
prediccion, probabilidad = frontPropagation(imagen)

print(prediccion, " con ", probabilidad)
