import numpy
from scipy.io import loadmat , savemat
from scipy.optimize import minimize

# conjunto entrenamiento
data = loadmat('digitos.mat')


X, y = data['X'], data['y'].ravel()
y[y == 10] = 0
m, n = X.shape

# configuracion de las capas
capa_entrada  = 400  # 20x20 entrada
capa_oculta = 25   # 25 unidades logisticas capa oculta
capa_salida = 10          # 10 etiquetas de salida

# funcion de activacion sigmoide
def sigmoide(h):
    return 1. / (1 + numpy.e ** (-h))

# inicializacion de los parametros de la red
def inicializarThetas(entrada, salida, epsilon_inic=0.12):
    t = numpy.zeros((salida, 1 + entrada))
    t = numpy.random.rand(salida, 1 + entrada) * 2 * epsilon_inic - epsilon_inic
    return t

# derivada de la funcion sigmoide
def derivada_sigmoide(z):
    g = sigmoide(z) * (1 - sigmoide(z))
    return g

# Computo funcion costo y Gradiente (implementacion de Back propagation)
def funcion_costo_gradiente(theta, capa_entrada, capa_oculta, capa_salida,X, y, lambda_=0.0):
    Theta1 = numpy.reshape(theta[:capa_oculta * (capa_entrada + 1)], (capa_oculta, (capa_entrada + 1)))
    Theta2 = numpy.reshape(theta[(capa_oculta * (capa_entrada + 1)):], (capa_salida, (capa_oculta + 1)))
    m = y.size
    J = 0
    Theta1_grad = numpy.zeros(Theta1.shape)
    Theta2_grad = numpy.zeros(Theta2.shape)

    # propagacion adelante para calcular h
    a1 = numpy.concatenate([numpy.ones((m, 1)), X], axis=1)
    a2 = sigmoide(a1.dot(Theta1.T))
    a2 = numpy.concatenate([numpy.ones((a2.shape[0], 1)), a2], axis=1)
    a3 = sigmoide(a2.dot(Theta2.T)) # h

    # convertimos componente de y en vector, ej: 3 en [0,0,0,1,0,0,0,0,0,0]
    y_vec = y.reshape(-1)
    y_vec = numpy.eye(capa_salida)[y_vec]

    temp1 = Theta1
    temp2 = Theta2

    # parametro de regularizacion
    reg_term = (lambda_ / (2 * m)) * (numpy.sum(numpy.square(temp1[:, 1:])) + numpy.sum(numpy.square(temp2[:, 1:])))
    # costo
    J = (-1 / m) * numpy.sum((numpy.log(a3) * y_vec) + numpy.log(1 - a3) * (1 - y_vec)) + reg_term

    # computo del gradiente (back propagation)
    # error de la ultima capa, lo que predecimos a3 menos lo que en realidad es
    delta_3 = a3 - y_vec
    # error en la capa 2
    delta_2 = delta_3.dot(Theta2)[:, 1:] * derivada_sigmoide(a1.dot(Theta1.T))

    # computamos propagacion del error en las capas respectivas
    Delta1 = delta_2.T.dot(a1)
    Delta2 = delta_3.T.dot(a2)

    # computo del gradiente con el parametro de  en cada capa
    Theta1_grad = (1 / m) * Delta1
    Theta1_grad[:, 1:] = Theta1_grad[:, 1:] + (lambda_ / m) * Theta1[:, 1:]

    Theta2_grad = (1 / m) * Delta2
    Theta2_grad[:, 1:] = Theta2_grad[:, 1:] + (lambda_ / m) * Theta2[:, 1:]

    # unimos en un vector
    grad = numpy.concatenate([Theta1_grad.ravel(), Theta2_grad.ravel()])

    return J, grad

# inicializamos los parametros
theta1_inicial = inicializarThetas(capa_entrada, capa_oculta)
theta2_inicial = inicializarThetas(capa_oculta, capa_salida)
Theta_inicial = numpy.concatenate([theta1_inicial.ravel(), theta2_inicial.ravel()], axis=0)

# preparamos la función para pasarla por el algoritmo de optimizacion
opciones = {'maxiter': 100}
# parametro de regularizacion (evitar el problema de sobre ajuste)
lambda_ = 0.1
# función para devolver J y Gradiente
funcion_costo_grad = lambda p: funcion_costo_gradiente(p, capa_entrada, capa_oculta, capa_salida, X, y, lambda_)
# algoritmo de optimizacion
res = minimize(funcion_costo_grad, Theta_inicial, jac=True, method='TNC', options=opciones)
# parametros optimos
parametros_optimos = res.x
# armamos segun topologia
Theta1 = numpy.reshape(parametros_optimos[:capa_oculta * (capa_entrada + 1)], (capa_oculta, (capa_entrada + 1)))

Theta2 = numpy.reshape(parametros_optimos[(capa_oculta * (capa_entrada + 1)):], (capa_salida, (capa_oculta + 1)))

# guardamos los parametro en
a1 = dict()
a2 = dict()
a1["Theta1"] = Theta1
a2["Theta2"] = Theta2

savemat("Theta1.mat", a1)
savemat("Theta2.mat", a2)
