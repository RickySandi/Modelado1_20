import numpy
from scipy.io import loadmat, savemat
import matplotlib.pyplot as pl
import cv2

def sigmoide(h):
    return 1. / (1 + numpy.e ** (-h))

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

def probarCalidadImagen(nomImagen):
    imagen = cv2.imread(nomImagen)
    imagenGris = cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY)
    imagenGris = cv2.GaussianBlur(imagenGris, (5,5), 0)
    ret, imagenBN = cv2.threshold(imagenGris, 90, 225, cv2.THRESH_BINARY_INV)
    cv2.imshow("extracts",imagenBN)
    cv2.waitKey()

def extraerEntradasPrueba (nomImagen):
    ### Limpieza
    imagen = cv2.imread(nomImagen)
    imagenGris = cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY)
    imagenGris = cv2.GaussianBlur(imagenGris, (5,5), 0)

    ret, imagenBN = cv2.threshold(imagenGris, 90, 225, cv2.THRESH_BINARY_INV)
    ### Encontrar Bordes
    grupos, i = cv2.findContours(imagenBN.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ventanas = [cv2.boundingRect(g) for g in grupos]
    ventanas = sorted(ventanas)
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
    return (numpy.array(Xs))

def predecirEntradas(nomImagen,thetas):
    reconocidos = []
    for i in extraerEntradasPrueba(nomImagen):
        x = i.reshape(1,-1)
        pred = frontPropagation(x,thetas[0],thetas[1])[-1].flatten()
        indMax = pred.argmax()
        idents = {10:0,11:"+",12:"-"}
        aux_pred = idents[indMax+1] if indMax+1>9 else indMax+1
        reconocidos+=[(aux_pred,pred[indMax])]
    return reconocidos

## Cargar Parametros
carga = loadmat('thetas.mat')
thetas=[carga["t1"]]
thetas+=[carga["t2"]]


## Imagen a predecir
nomImagen = "operaciones/pruebaTodos.jpg"

## Ver si la imagen es viable
probarCalidadImagen(nomImagen)

## Predecir las entradas
reconocidos = predecirEntradas(nomImagen,thetas)
print("prediccion, certeza: ", reconocidos)

'''#### Resolver operacion leida
ops = {11:(lambda x,y: x+y),12:(lambda x,y: x-y)}
p1 = reconocidos[0][0]
i=1
while(i<len(reconocidos)):
  op = ops[reconocidos[i][0]]
  p2 = reconocidos[i+1][0]
  p1 = op(p1,p2)
  i+=2
print("El resultado de las operaciones = ",p1)'''

