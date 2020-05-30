import numpy
import cv2
from scipy.io import loadmat, savemat

def extraerEntradasPrueba (nomImagen):
    ### Limpieza
    imagen = cv2.imread(nomImagen)
    imagenGris = cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY)
    imagenGris = cv2.GaussianBlur(imagenGris, (5,5), 0)

    ret, imagenBN = cv2.threshold(imagenGris, 90, 225, cv2.THRESH_BINARY_INV)
    ### Encontrar Bordes
    grupos, i = cv2.findContours(imagenBN.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ventanas = [cv2.boundingRect(g) for g in grupos]
    ### Encuadre
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
            #cv2.imshow("d", digito)
            #cv2.waitKey()
            Xs += [digito.T.flatten()]
        indice+=1
    cv2.imshow("Digitos", imagen)
    cv2.waitKey()
    return (numpy.array(Xs))

sumas = extraerEntradasPrueba("sumas1.jpg")
restas = extraerEntradasPrueba("restas.jpg")
print("sumas y restas: ",sumas.shape,restas.shape)
ys = numpy.array([(lambda i: 11 if(i<sumas.shape[0]) else 12)(i) for i in range(sumas.shape[0]+restas.shape[0])])
xs = numpy.append(sumas, restas, axis= 0)
print("entradas y etiquetas: ",xs.shape,ys.shape)
data = loadmat("digitos.mat")
digitos = data["X"]
digitosEtiq = data["y"]
xs = numpy.append(digitos,xs, axis = 0)
ys = ys.reshape(-1,1)
ys = numpy.append(digitosEtiq,ys, axis = 0)

guardado = dict()
guardado["X"] = xs
guardado["y"] = ys
savemat("Entradas.mat",guardado)