import numpy
from scipy.io import loadmat, savemat
import cv2

imagen = cv2.imread("mas.jpg")
imagenGris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
imagenGris = cv2.GaussianBlur(imagenGris, (5, 5), 0)
ret, imagenBN = cv2.threshold(imagenGris, 90, 255, cv2.THRESH_BINARY_INV)
grupos, _ = cv2.findContours(imagenBN.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

digitos_num = loadmat('digitos.mat')
X = digitos_num['X']
y = digitos_num['y']

ventanas = [cv2.boundingRect(g) for g in grupos]

# rutina para adjuntar simbolo + al conjunto entrenamiento
for g in ventanas:
    cv2.rectangle(imagen, (g[0], g[1]), (g[0] + g[2], g[1] + g[3]), (255,0,0), 2)
    l = int(g[3] * 1.6)
    p1 = abs(int(g[1] + g[3] // 2) - l // 2)
    p2  = abs(int(g[0] + g[2] // 2) - l // 2)

    digito = imagenBN[p1: p1+l, p2: p2+l]

    digito = cv2.resize(digito, (20, 20), interpolation=cv2.INTER_AREA)
    digito = cv2.dilate(digito, (3, 3,))
    aux = digito.T.flatten()  # vector 400 posiciones

    X = numpy.append(X, aux.reshape(1, 400), axis=0)
    y = numpy.append(y, numpy.array([11])).flatten()

    cv2.imshow("capturando", digito)
    cv2.waitKey()

    dicAux = dict()
    dicAux["X"] = X
    dicAux["y"] = y.flatten()

cv2.waitKey()
savemat("digitosNuevo.mat", dicAux)

