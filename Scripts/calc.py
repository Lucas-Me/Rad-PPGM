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
	Calcula a derivada da função Y com relaçao a X, utilizando
	o método trapezoidal. 

	Entrada: Array

		y = array de N elementos com os resultados da função em cada X.
		x = array de N elementos com o respectivo valor X.

	Saída: Float
	'''

	# Calculando a area de cada intervalo (integral)
	dx = x[1:] - x[:-1]
	area = dx * (y[1:] + y[:-1]) / 2

	# Somando as areas, equivale a integral.
	return np.nansum(area)


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


def density_water_vapor(T, Td, p, w):
	'''
	Calcula a densidade de vapor d'água na parcela, dada a pressão, temperatura,
	razão de mistura e temperatura do ponto de orvalho.

	Entrada: Array | float

		T: Temperatura [K]
		Td: Temperatura do ponto de orvalho [K]
		p: Pressão atmosférica [hPa]
		w: razão de mistura [Kg / Kg]

	Saída: Array | float - [Kg / m³]
	'''

	# Calculando a pressão de vapor atual da parcela
	# --------------------------------------------------------
	e = saturation_vapor_pressure(Td - 273.15)

	# Calculando a densidade do vapor d´agua
	# --------------------------------------------------------
	# Constante do ar seco
	R = 287 # [J * kg^-1 * K^-1] 

	# densidade do ar seco [Kg * hPa * J^-1]
	Qd = (p - e) / (R * T)

	# Converte unidade para Kg / m³
	Qd = Qd * 100

	# densidade do vapor d´agua - Qv / Qd = w
	Qv = w * Qd

	return Qv


def path_length(Qv, z):
	'''
	Calcula o Path Length, definida como uma integral da densidade do gás 
	absorvedor em função da altitude, desde o topo da atmosfera até o nível de
	interesse. (Equação 3.2.16 do livro do LIOU)

	Porém, para aplicação no IR termal, consideramos da superfície para cima,
	de modo que u(z = 0) = 0.

	Entrada: Array

		Qv: Densidade do gás absorvedor em cada nível z. [g / cm³ ou Kg / m³]
		z: array de nível z [cm ou m]

	Saída: Array  [g / cm² ou Kg / m²]
	'''
	
	# CONSIDERANDO QUE Z ESTEJA EM ORDEM CRESCENTE
	resultados = np.zeros(z.shape, dtype = np.float64)
	for i in range(resultados.shape[0]):
		resultados[i] = integral(Qv[:i + 1], z[:i + 1])

	return resultados

def planck(_lambda, T):
	'''
	Calcula a lei de Planck para um dado comprimento de onda e temperatura.

	Entrada: Array | Float

		_lambda: Comprimento de onda [m]
		T: Temperatura [K]

	Saída: Array | Float - [J * m^-3 * s^-1]
	'''

	# Constante de Planck [J * s]
	H = 6.626 * 1e-34

	# Constante de Boltzmann (J / K)
	K = 1.3806 * 1e-23 

	# Velocidade da luz [m/s]
	c = 3 * 1e8

	# Lei de Planck
	numerador = 2 * H * c ** 2 # [J * m² / s]
	denominador = _lambda ** 5 * (np.exp(H * c / (K * _lambda * T)) - 1) # [m^5]
	
	# Resultado [J * m^-3 * s^-1]
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
