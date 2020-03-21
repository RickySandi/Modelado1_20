# from scipy.io import loadmat

from Logistica import *
from scipy.io import loadmat
import cv2 #pip3 install python-opencv

l = Logistica("digitos.mat", 0.1)

print(l.X.shape)
print(l.y.shape)
print(l.Theta.shape)

l.verDigito(l.X[4456, 1:])
print(l.y[4456, :])
#
demo = numpy.zeros(401)
print(l.funcionCosto(demo, 8))
print(l.gradiente(demo, 8).shape)

l.unoVsResto()
print(l.Theta.shape)

#l.predecirNumero("#vector#")


l.predecirNumero("test/test03/jpg")

param = loadmat("Thetas.mat") #cargando modelo
# entrenando 
x = l.X[1345, :].reshape(401, 1)
l.verDigito(l.X[1345,1:])
print("Lo que es: ", l.y[1345])
prediccion = l.sigmoide(param["Theta"].dot(x))
num_predic = prediccion.argmax() + 1
print("lo que el modelo dice: ", num_predic)
print(" con ", prediccion[num_predic] * 100, "% de certeza")

# #prueba con los jpg 
# imagen1 = cv2.imread("test/test01/jpg")
# print(imagen1.shape)
# imagenGris = cv2.cvtColor(imagen1,cv2.COLOR_RGB2GRAY)
# print(imagenGris.shape)

# captura = cv2.VideoCapture(0)
# while True:
#     imagen, frame = captura.read()
#     cv2.imshow("imagen", frame)
#     if cv2.waitKey(50) >=0:
#         break
