# from scipy.io import loadmat

from Logistica import *
from scipy.io import loadmat
import cv2 #pip3 install python-opencv

# l = Logistica("digitos.mat",0.1)
# l.unoVsResto()
# l.cargarParametros()
# # print(l.Theta.shape)

# # x = l.X[4000,1:]
# # l.verDigito(x)
# # print(l.predecirNumero(x))

# print(l.predecirNumero("test23.jpg"))

img2 = cv2.imread("numerosAMano.jpeg")
img2 = cv2.cvtColor(img2,cv2.COLOR_RGB2GRAY)
img2 = cv2.GaussianBlur(img2,(5,5),0) #Suavizar contornos

valor,img2 = cv2.threshold(img2,90,255,cv2.THRESH_BINARY_INV) # convertir blanco y negro 
pos, i = cv2.findContours(img2.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #Lista de contornos 
grupos = [cv2.boundingRect(g) for g in pos]
print(grupos)

for g in grupos:
    cv2.rectangle((img2,g[0],g[1]),((img2,g[0],g[1])))
