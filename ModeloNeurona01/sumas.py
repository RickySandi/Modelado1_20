# import sys 

# sys.path.append('/usr/local/lib/python3.7/site-packages') 
import numpy
import cv2
from scipy.io import loadmat

imagen = cv2.imread("menos.jpg")
imagenGris = cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY)
imagenGris = cv2.GaussianBlur(imagenGris, (5,5), 0)

ret, imagenBN = cv2.threshold(imagenGris, 90, 225, cv2.THRESH_BINARY_INV)

grupos, i = cv2.findContours(imagenBN.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
ventanas = [cv2.boundingRect(g) for g in grupos]


indice = 0
Xs = []
for g in ventanas:
    #print(indice)
    cv2.rectangle(imagen, (g[0], g[1]), (g[0]+g[2], g[1]+g[3]),(255,0,0),2)
    l = int(g[3] * 1.6)
    p1 = int(g[1] + g[3] // 2) - l // 2
    p2 = int(g[0] + g[2] // 2) - l // 2
    digito = imagenBN[p1: p1+l, p2: p2+l]
    if(digito.shape[1] != 0 and digito.shape[0] != 0 ):
        #print(digito.shape)
        digito = cv2.resize(digito, (20,20), interpolation = cv2.INTER_AREA)
        digito = cv2.dilate(digito, (3,3))
        cv2.imshow("d", digito)
        cv2.waitKey()
        Xs += [digito.flatten()]
    indice+=1
Xs = numpy.array(Xs)

print(Xs.shape)
cv2.imshow("Digitos", imagen)
cv2.waitKey()