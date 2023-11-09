# ===================================================================
# 				Script com funções mais simples que calculam
#  	      	derivada, integral, densidade do gás, path length
# 				Lei de Planck e Lei de Stefan - Boltzmann

# Made by: Lucas da Silva Menezes
# 31/07/2023
# ===================================================================

# IMPORTS
import numpy as np


def integral(y, x):
	'''
	Calcula a integral da função Y com relaçao a X, utilizando
	o método trapezoidal. 

	Entrada: Array

		y = array de N elementos com os resultados da função em cada X.
		x = array de N elementos com o respectivo valor X.

	Saída: Float
	'''

	# Calculando a area de cada intervalo (integral)
	dx = np.diff(x)
	area = dx * (y[1:] + y[:-1]) / 2

	# Somando as areas, equivale a integral.
	return np.nansum(area)

def non_linear_derivative(y, x):
	'''
	Calcula a derivada da função Y com relaçao a X, utilizando
	o método de diferenca centradas em uma grade irregular.

	Parameters
	----------
	x: array[float] - shape[N]
		Pontos de grade x
	y: array[float] - shape[N]
		Pontos de grade em y
	
	Returns
	-------
		dydx : array[float] - shape[N - 2]
	'''
	
	dydx = []
	for i in range(1, y.shape[0] - 1):
		# Método 1
		# ---------
		# parametros
		h_minus = x[i] - x[i - 1]
		h_plus = x[i + 1] - x[i]
		y_minus = y[i - 1]
		y_0 = y[i]
		y_plus = y[i + 1]

		# calculando termos
		termo1 = h_plus / (h_plus + h_minus) * (y_plus - y_0) / h_plus
		termo2 = h_minus / (h_plus + h_minus) * (y_0 - y_minus) / h_minus

		# armazena o resultado
		result = termo1 + termo2

		# Método 2
		# ---------
		# parametros
		# h_plus = x[i + 1] - x[i]
		# h_minus = x[i] - x[i - 1]

		# # calcula os termos
		# termo1 = (1 / h_plus - 1 / (h_plus + h_minus)) * y[i + 1]
		# termo2 = (1 / h_minus - 1 / h_plus) * y[i]
		# termo3 = (1 / (h_plus + h_minus) - 1 / h_minus) * y[i - 1]

		# # armazena o resultado
		# result = termo1 + termo2 + termo3		

		# Método 3
		# ---------
		# # parametros
		# h_plus = x[i + 1] - x[i]
		# h_minus = x[i] - x[i - 1]

		# # calcula os termos
		# result = (y[i + 1] - y[i - 1]) / (h_plus + h_minus)

		dydx.append(result)

	return np.array(dydx)


def first_derivative(y, x):
	'''
	Calcula a derivada da função Y com relaçao a X, utilizando
	uma mistura do metodo central, forward e backward.

	Parameters
	----------
	x: array[float] - shape[N]
		Pontos de grade x
	y: array[float] - shape[N]
		Pontos de grade em f
	
	Returns
	-------
		dydx : array[float] - shape[N]
	'''

	# inicializa a variavel
	# dydx = np.full(y.shape[0], np.nan)
	
	# # Forward
	# dydx[0] = (y[1] - y[0]) / (x[1] - x[0])

	# # Backward
	# dydx[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])

	# # Method A - "Playing with nonuniform grids", Veldman & Rinzema (1992)
	# dydx[1:-1] = non_linear_derivative(y, x)

	dydx = np.gradient(y, x)

	return dydx


def mixing_ratio_from_dewpoint_pressure(Td, p):
	'''
	Calcula a razão de mistura a partir da temperatura do ponto de orvalho
	e pressão atmosférica

	OBS: Se Td for passado na equação da pressão de vapor de saturação, o resultado
	será a pressão de vapor atual da parcela.

	Parameters
	----------
	Td: float | array[float]
		Temperatura do ponto de orvalho [K]
	p: float | array[float]
		pressão atmosférica [hPa]
	'''

	# Calcula a  pressão de vapor
	e = saturation_vapor_pressure(Td - 273.15)

	# calcula a razão de mistura
	w = 621.97 * e / (p - e)

	return w

def saturation_vapor_pressure(T):
	'''
	Calcula a pressão de vapor de saturação necessária para uma parcela de ar à temperatura
	T estar saturada.

	OBS: Caso seja passado Td (ponto de orvalho), o resultado será a pressão de vapor
	atual da parcela.

	Entrada: Array | float

		T : Temperatura [Celsius] 

	Saída: Array | Float - [hPa] ou [mb] (são a mesma coisa)
	'''

	return 6.112 * np.exp((17.67 * T) / (243.5 + T))

def relative_humidty_from_density(Qv, T):
	'''
	calcula a umidade relativa da parcela de ar, dado a densidade do vapor d'agua
	e a temperatura da parcela.

	Entrada: Array[Float] | Float

		Qv : Densidade do vapor d'agua [Kg / m^3]
		T : Temperatura [K]

	Saída: Array[Float] | Float  - [adimensional]
	'''

	# Constante individual do vapor d'agua
	Rv = 461.5 #  [J / (Kg * K)]

	# Pressao de vapor da parcela SATURADA [hPa]
	es = saturation_vapor_pressure(T - 273.15)
	es = es * 1e2 # converte para [Pa]

	# Densidade do vapor d'água na parcela SATURADA
	Qvs = es / (Rv * T)

	# Umidade relativa da parcela
	ur = Qv / Qvs

	return ur

def density_water_vapor(T, Td):
	'''
	Calcula a densidade de vapor d'água na parcela, dada a temperatura e temperatura do ponto de orvalho.

	Entrada: Array | float

		T: Temperatura [K]
		Td: Temperatura do ponto de orvalho [K]

	Saída: Array | float - [Kg / m³]
	'''

	# Calculando a pressão de vapor atual da parcela
	# --------------------------------------------------------
	e = saturation_vapor_pressure(Td - 273.15)

	# Calculando a densidade do vapor d´agua
	# --------------------------------------------------------
	# Constante individual do vapor d'agua
	Rv = 461.5# [J * kg^-1 * K^-1] 

	# densidade do vapor d'agua [Kg * hPa * J^-1]
	Qv = e / (Rv * T)

	# # Converte unidade para Kg / m³
	Qv = Qv * 100

	return Qv

def water_vapor_density_from_humidity(T, ur):
	'''
	calcula a densidade do vapor d'agua na parcela de ar a partir da temperatura e umidade relativa.

	Entrada: Array[Float] | Float

		ur : Umidade relativa [adimensional]
		T : Temperatura [K]

	Saída: Array[Float] | Float  - [Kg / m³]
	'''

	# Constante individual do vapor d'agua
	Rv = 461.5 #  [J / (Kg * K)]

	# Pressao de vapor da parcela SATURADA [hPa]
	es = saturation_vapor_pressure(T - 273.15)
	es = es * 1e2 # converte para [Pa]

	# Densidade do vapor d'água na parcela SATURADA
	Qvs = es / (Rv * T)

	# Umidade relativa da parcela
	Qv = ur * Qvs

	return Qv

def path_length(Qv, z):
	'''
	Calcula o Path Length, definida como uma integral da densidade do gás 
	absorvedor em função da altitude, desde o topo da atmosfera até o nível de
	interesse. (Equação 3.2.16 do livro do LIOU)

	U(z = infinito) = 0

	Entrada: Array

		Qv: Densidade do gás absorvedor em cada nível z. [g / cm³ ou Kg / m³]
		z: array de nível z [cm ou m]

	Saída: Array  [g / cm² ou Kg / m²]
	'''
	
	# CONSIDERANDO QUE Z ESTEJA EM ORDEM CRESCENTE
	resultados = np.zeros(z.shape, dtype = np.float64)
	for i in range(resultados.shape[0]):
		resultados[i] = integral(Qv[i:], z[i:])

	return resultados

def planck(v, T):
	'''
	Calcula a lei de Planck para um dado comprimento de onda e temperatura.

	Entrada: Array | Float

		v: frequência [1/s]
		T: Temperatura [K]

	Saída: Array | Float - [J * m^-2]
	'''

	# Constante de Planck [J * s]
	H = 6.626 * 1e-34

	# Constante de Boltzmann (J / K)
	K = 1.3806 * 1e-23 

	# Velocidade da luz [m/s]
	c = 3 * 1e8

	# Lei de Planck
	numerador = 2 * H * v ** 3 # [J * s^-2]
	denominador = c ** 2 * (np.exp(H * v / (K * T)) - 1) # [m² / s²]
	
	# Resultado [J * m^-2]
	B = numerador / denominador

	return B

def planck_k(k, T):
	'''
	Calcula a lei de Planck para um dado numero de onda e temperatura.

	Parameters
	----------
	k: float | array[float]
		Numero de onda [1 / m]
	T: float | array[float]
		Temperatura [K]

	Returns
	-------
	B: float | array[float]
		radiancia monocromatica [energy/area/time/sr/frequency]
	'''

	# Constante de Planck [J * s]
	H = 6.626 * 1e-34

	# Constante de Boltzmann (J / K)
	K = 1.3806 * 1e-23 

	# Velocidade da luz [m/s]
	c = 3 * 1e8

	# Frequencia [1 / s]
	v = c * k

	# Lei de Planck
	numerador = 2 * H * v ** 3 # [J * s^-2]
	denominador = c ** 2 * (np.exp(H * v / (K * T)) - 1) # [m² / s²]
	
	# Resultado [J * m^-2]
	B = numerador / denominador

	return B


def stefan_boltzmann(T):
	'''
	Calcula a Lei de Stefan-Boltzmann para um dada Temperatura.

	Entrad: Array | Float
	
		T: Temperatura [k]

	Saída: Array | Float - [J * m^-2 * s^-1]
	'''

	# Constante
	CTE = 5.67 * 1e-8 # [J * m^-2 * s^-1 * K*^-4]

	return CTE * T ** 4
