# ===================================================================
# 				Script com funções  mais complexas
# 			relacionadas ao do cooling rate considernado
#				uma atmosfera com nuvens e sem nuvens.


# Referencias utilizadas
#	[1] Parametrization of Infrared Radiative Transfer in Cloudy Atmospheres
# 		(Liou, 1981)
		

# Made by: Lucas da Silva Menezes
# 02/08/2023
# ===================================================================


# IMPORTS
# -------
import numpy as np

# IMPORTS LOCAIS
# --------------
import scripts.broadband as broadband
import scripts.calc as calc


class CoolingRate(object):
	'''
	Classe responsável por armazenar os parâmetros e resultados relacionados
	ao cálculo da taxa de resfriamento considerando apenas o efeito do vapor d'água
	na banda do Infravermelho e a abordagem plano-paralelo.

	Neste abordagem, u(z = 0) = 0, e no topo da atmosfera u = u1.
	Sendo u1 o path length total na coluna atmosférica.
	'''
	
	# Constantes
	Cp = 1005 # Calor específico a pressão cte [J * Kg^-1 * K^-1]
	Rv = 461.5 # Cte individual do vapor d'agua [J * Kg^-1 * K^-1]

	def __init__(self, T, u, q, p, Qv):
		'''
		Parameters
        ----------
		T : Array[float]
			A temperatura do ar [K]
		u : Array[float]
			Path Length [g / cm²]
		q : Array[float]
			Razão de mistura do ar úmido [adimensional - Kg/Kg]
		p : Array[float]
			Pressão atmosférica [hPa]
		Qv : Array[float]
			Densidade do vapor d'água na parcela [Kg / m³] 
		'''

		# Define as variáveis privadas desta classe
		self.T = T
		self.u = u if u[1] - u[0] > 0 else u[0] - u # Garante que "u" é crescente
		self.q = q
		self.p = p
		self.Qv = Qv

		# Calcula parâmetros que serão utilizados, baseado nas variaveis acima
		# ---------------------------------------------------------------------
		# Pressão parcial do vapor na parcela de ar [Pa]
		self.e = Qv * self.Rv * T  # Lei do gás ideal 

		# Path length (u) corrigido para cada banda de absorcao
		self.u_rot = broadband.path_length_rot(T, p, u)
		self.u_cont = broadband.path_length_cont(T, u, self.e)
		self.u_vib = broadband.path_length_vib(u, p)

	def clear_sky(self, band : str  = 'all'):
		'''
		Taxa de resfriamento considerando um céu limpo, sem nuvens.

		Parameters
		----------
		band : str
			Especifica a banda de absorcao a ser considerada, default = todas.
			Opcoes disponiveis são:
			- 'all' : todas as bandas
			- 'rot' : apenas a banda do rotacional.
			- 'cont' : apenas a banda do continuum.
			- 'vib' : apenas a banda do vibracional-rotacional.

		Returns
		-------
		cooling_rate : array[float]
			Taxa de resfriamento [Kelvin / dia]
		'''

		# Termo 1: Sigma T^4 * dEf(u1 - u, T1)/du 
		# -------------------------------------------
		T_topo = self.T[-1] # "-1" indica no topo da atmosfera

		# emissividade em cada nivel
		Ef = [self._emissivity(
			T_topo, 
			n1 = i, # "U" em toda a coluna
			n2 = self.u.shape[0] - 1,
			u_rot = self.u_rot[-1],
			u_vib = self.u_vib[-1],
			band = band
			) for i in range(self.u.shape[0])]
		
		# Derivada simples (u está em ordem crescente)
		du = np.diff(self.u) #  [g / cm²]
		dEf = np.diff(np.array(Ef))
		dEf1du = dEf / du # [( g / cm²)^-1]

		# Resultado do termo 1
		termo1 = -1 * calc.stefan_boltzmann(T_topo) * dEf1du
		termo1 = np.append(termo1, np.nan) # [J * m^-2 * s^-1 * ( g / cm²)^-1]
		termo1 = termo1 / 10 # corrigindo unidade para [J * s^-1 * Kg^-1]

		# Termo 2: Integral
		# -------------------------------------------

		# inicializa o termo
		termo2 = np.full(self.u.shape, fill_value=np.nan, dtype = np.float64)

		# termos medios
		u_mean = (self.u[:-1] + self.u[1:]) / 2
		T_mean = np.interp(u_mean, self.u, self.T)
		u_rot_mean = np.interp(u_mean, self.u, self.u_rot)
		u_vib_mean = np.interp(u_mean, self.u, self.u_vib)

		# d(Sigma * T(u')^4)/du'
		du_ = np.diff(self.u)
		dsigma_t_4_du = np.diff(calc.stefan_boltzmann(self.T)) / du_

		# loop para cada nível da sondagem
		for i in range(termo2.shape[0] - 1): # ultimo nao incluso
			
			# Função a ser integrada
			dEfdu = np.full(du_.shape[0], fill_value=np.nan)

			# loop para  a integral, da superfície até o topo da sondagem
			for j in range(du_.shape[0]):
				# emissividade em cada nivel
				Ef2 = self._emissivity(
					T_mean[j],
					n1 = i + 1,
					n2 = j,
					u_rot = u_rot_mean[j],
					u_vib = u_vib_mean[j],
					band = band
					)
				
				Ef1 = self._emissivity(
					T_mean[j],
					n1 = i,
					n2 = j,
					u_rot = u_rot_mean[j],
					u_vib = u_vib_mean[j],
					band = band
				)

				# Derivada simples (u está em ordem crescente)
				dEf = Ef2 - Ef1
				dEfdu[j] = dEf / du_[i]

			# Função a ser integrada
			f = dEfdu * dsigma_t_4_du
			termo2[i] = np.nansum(f * du_) / 10 # Conserta a unidade

		# Resultado final
		cooling_rate = -1 * (termo1 + termo2) * self.q / self.Cp # [K/s]

		# Inverte o array e converte a unidade para [K / day]
		cooling_rate = cooling_rate * 3600 * 24

		return cooling_rate

	def _emissivity(self, T, n1, n2, u_rot, u_vib, band : str = 'all'):
		'''
		Calcula a emissividade broadband, considerando todas as bandas ou apenas
		a banda de absorcao especificada pelo usuario.
		Considera o path length do nivel inicial (base) ate um outro nivel final
		da camada atmosferica.

		Parameters
		----------
		T : float
		 	Temperatura do ar [K]
		n1 : int
			Indice que representa o nivel inicial da coluna atmosferica
		n2 : int
			Indice que representa o nivel final da coluna atmosferica.
		u_rot : float
			Termo do u corrigido para o rotacional em n2, ou o termo medio
		u_vib : float
			Termo do u corrigido para o vibracional em n2, ou o termo medio
		band : str
			Especifica a banda de absorcao a ser considerada, default = todas.
			Opcoes disponiveis são:
			- 'all' : todas as bandas
			- 'rot' : apenas a banda do rotacional.
			- 'cont' : apenas a banda do continuum.
			- 'vib' : apenas a banda do vibracional-rotacional.

		Returns
		-------
		emissivity : float
			Emissividade broadband [adimensional]
		'''

		# Caso 1: Todas as bandas (rotational, continuum, vibrational-rotational)
		if band == 'all':
			v1, tau1 = broadband.transmitance_rot(
				T,
				np.abs(self.u_rot[n1] - u_rot)
				)
			v2, tau2 = broadband.transmitance_cont(
				T,
				self.u_cont[n1:n2 + 1],
				self.p[n1:n2 + 1],
				self.e[n1:n2 + 1]
				)
			v3, tau3 = broadband.transmitance_vib(
				np.abs(self.u_vib[n1] - u_vib)
				)

			intervalos = np.concatenate((v1, v2, v3)) # Intervalos de numero de onda
			transmitance = np.concatenate((tau1, tau2, tau3)) # Transmitancia em cada intervalo
		
		# Caso 2:
		elif band == 'rot':
			intervalos, transmitance = broadband.transmitance_rot(
				T,
				np.abs(self.u_rot[n1] - u_rot)
				)

		# Caso 3: 
		elif band == 'cont':
			intervalos, transmitance = broadband.transmitance_cont(
				T,
				self.u_cont[n1:n2 + 1],
				self.p[n1:n2 + 1],
				self.e[n1:n2 + 1]
				)

		# Caso 4:
		elif band == 'vib':
			intervalos, transmitance = broadband.transmitance_vib(
				np.abs(self.u_vib[n1] - u_vib)
				)
		
		else: # Lança um erro
			raise KeyError
		
		# Lei de Stefan-Boltzmann
		steboltz = calc.stefan_boltzmann(T) # J * m^-2 * s^-1

		# transforma em m^-1 e converte para frequencia (1/s)
		c = 3 * 1e8
		f = c * (intervalos * 100)
		fmean = np.mean(f, axis = 1) # [1/s]
		df = np.abs(f[:, 1] - f[:, 0]) # [1/s]

		# Calcula a emissividade de cada intervalo da integral (freq) e soma
		emissivity = np.nansum(
			np.pi * calc.planck(fmean, T) * (1 - transmitance) * df / steboltz
			)
		
		return emissivity