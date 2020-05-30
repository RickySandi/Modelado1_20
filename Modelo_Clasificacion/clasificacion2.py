import numpy
from scipy.io import loadmat, savemat
import cv2

t1 = loadmat("Theta1.mat")["Theta1"]
t2 = loadmat("Theta2.mat")["Theta2"]

def frontPropagation(imagen, t1, t2):
    m, n = imagen.shape
    a1 = imagen  # capa 1
    aux = numpy.ones((n+1, m))
    aux[1:, :] = imagen.T
    a1 = aux  # 401x5000

    # t1 (25x401) a1 (401x5000) -> a2(25 x 5000)
    a2 = t1.dot(a1)  # 25*m
    a2 = sigmoide(a2)

    aux = numpy.ones((26, m))
    aux[1:, :] = a2
    a2 = aux  # a2 (26x5000)
    # t2 (10x26)
    a3 = t2.dot(a2)  # 10*26 prediccion
    a3 = sigmoide(a3)
    h = a3 # 10 x 5000
    return h.argmax(),

def sigmoide(h):
    return 1. / (1 + numpy.e ** (-h))


camara = cv2.VideoCapture(0)

while True:
    imagen, escena = camara.read()
    imagenGris = cv2.cvtColor(escena, cv2.COLOR_BGR2GRAY)
    imagenGris = cv2.GaussianBlur(imagenGris, (5,5), 0)
    ret, imagenBN = cv2.threshold(imagenGris, 90, 255, cv2.THRESH_BINARY_INV)
    grupos, _ = cv2.findContours(imagenBN.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ventanas = [cv2.boundingRect(g) for g in grupos]
    for g in ventanas:
        cv2.rectangle(imagen, (g[0], g[1]), (g[0] + g[2], g[1] + g[3]), (255, 0, 0), 2)
        l = int(g[3] * 1.6)
        p1 = int(g[1] + g[3] // 2) - l // 2
        p2 = int(g[0] + g[2] // 2) - l // 2
        if p1 > 0 and p2 > 0:
            digito = imagenBN[p1: p1 + l, p2: p2 + l]

        digito = cv2.resize(digito, (20, 20), interpolation=cv2.INTER_AREA)
        digito = cv2.dilate(digito, (3, 3,))
        aux = digito.T.flatten()  # vector 400 posiciones
        aux = aux.reshape(aux.size, 1)

        prediccion = frontPropagation(aux.T, t1, t2)
        cv2.putText(imagen, str(prediccion[0]), (g[0], g[1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
        print(prediccion[0], " con ")
    cv2.imshow('Prueba', imagenBN)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
camara.release()
cv2.destroyAllWindows()