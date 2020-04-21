
import numpy
import matplotlib.pyplot as graf
from scipy.optimize import fmin_bfgs
from scipy.io import loadmat, savemat
import scipy.misc
import cv2 


class Logistica:

    def __init__(self, archivo="", lambdaReg=0):
        numpy.seterr(all="ignore")  # ignorar los errores en numpy
        self.X = None
        self.y = None
        self.Theta = None
        if archivo != "":
            self.cargarConjunto(archivo)
            self.lambdaReg = lambdaReg

    def cargarConjunto(self, archivo):

        datos = loadmat(archivo)
        self.X = datos['X']
        self.y = datos['y']

        m, n = self.X.shape
        self.X = numpy.append(numpy.ones((m, 1)), self.X, axis=1)
        self.Theta = numpy.ones(n+1)
    
    def cargarParametros(self):
        self.Theta = loadmat("Thetas.mat")["Theta"] #401x1


    def sigmoide(self, h):
        return 1. / (1 + numpy.e ** (-h))


    def funcionCosto(self, theta, etiqueta):
        theta = theta.reshape(theta.size, 1)
        y = (self.y == etiqueta)
        m = self.y.shape[0]
        h = self.sigmoide(self.X.dot(theta))
        parametroRegul = (self.lambdaReg / m) * (numpy.power(self.Theta, 2).sum())
        J = (1. / m) * (-y.transpose().dot(numpy.log(h)) - (1. - y).transpose().dot(numpy.log(1. - h)))
        J = J.sum() + parametroRegul
        return J

    def funcionCostoVector(self, theta, etiqueta):
        theta = theta.reshape(theta.size, 1)
        y = (self.y == etiqueta)
        m = self.y.shape[0]
        h = self.sigmoide(self.X.dot(theta))
        parametroRegul = (self.lambdaReg / m) * (numpy.power(self.Theta, 2).sum())
        J = (1. / m) * (-y.transpose().dot(numpy.log(h)) - (1. - y).transpose().dot(numpy.log(1. - h)))
        #J = J.sum() + parametroRegul
        return J

    def gradiente(self, theta, etiqueta):
        theta = theta.reshape(theta.size, 1)
        y = (self.y == etiqueta)
        m = self.y.shape[0]
        h = self.sigmoide(self.X.dot(theta))
        grad = (1. / m) * self.X.transpose().dot(h - y)
        parametroRegul = ((self.lambdaReg / m) * self.Theta)
        parametroRegul = parametroRegul.reshape(parametroRegul.size, 1)
        grad = grad + parametroRegul
        return grad.flatten()

    def entrenar(self, theta, etiqueta):
        self.Theta = fmin_bfgs(self.funcionCosto, theta, fprime=self.gradiente, args=(etiqueta, ))

    def unoVsResto(self):
        thetaInicial = self.Theta
        thetaResul = numpy.ones((10, self.Theta.size))
        for i in range(1, 11):
            self.entrenar(thetaInicial, i)
            thetaResul[i-1, :] = self.Theta

        a = dict()
        a["Theta"] = thetaResul
        savemat("Thetas.mat", a)
        self.Theta = thetaResul


    @staticmethod
    def verDigito(digito):

        m = digito.reshape(20, 20)
        graf.imshow(m.T, cmap='Greys_r')
        graf.show()

    def predecirNumero(self, imagen=""):

       # param = loadmat("Thetas.mat")
       # l = Logistica("digitos.mat", 0.1)
        imagen = cv2.imread(imagen)
       # print(type(imagen))
        imagen = cv2.cvtColor(imagen,cv2.COLOR_RGB2GRAY)
        # print(imagenGris.shape)

        imagen = imagen.flatten()
        imagen = numpy.append(numpy.ones(1),imagen.reshape(1,400))
        print(imagen.shape)
        imagen = imagen.reshape(401,1)

        #print("Lo que es: ", imagen)
        prediccion = self.sigmoide(self.Theta.dot(imagen))
        num_predic = prediccion.argmax() + 1
        # print("lo que el modelo dice: ", num_predic)
        # print(" con ", prediccion[num_predic] * 100, "% de certeza")

        return num_predic







