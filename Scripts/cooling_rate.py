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
import scipy.interpolate as interp

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

		# Numero de niveis
		self.nlevs = self.u.shape[0]

		# Calcula parâmetros que serão utilizados, baseado nas variaveis acima
		# ---------------------------------------------------------------------
		# Pressão parcial do vapor na parcela de ar [Pa]
		self.e = Qv * self.Rv * T  # Lei do gás ideal 

		# Path length (u) corrigido para cada banda de absorcao
		self.u_rot = broadband.path_length_rot(T, p, self.u)
		self.u_cont = broadband.path_length_cont(T, self.u, self.e)
		self.u_vib = broadband.path_length_vib(self.u, p)

		# Derivadas importantes
		# ---------------------------------------------------------------------
		self.dSigmaT4du = calc.first_derivative(
			calc.stefan_boltzmann(self.T), self.u)
		print(self.dSigmaT4du)


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
		Ef = np.array([self._emissivity(
			T_topo, 
			n1 = self.u.shape[0] - 1, # "U" em toda a coluna
			n2 = i,  
			u_vib = self.u_vib[-1],
			band = band
			) for i in range(self.nlevs)])
		
		# Derivada simples (u está em ordem crescente)
		# dEf1du = calc.first_derivative(Ef, self.u)
		du = np.diff(self.u) #  [g / cm²]
		dEf1du = np.diff(Ef) / du
		dEf1du = np.append(dEf1du, np.nan)

		# Resultado do termo 1
		termo1 = -1 * calc.stefan_boltzmann(T_topo) * dEf1du
		termo1 = termo1 / 10 # corrigindo unidade para [J * s^-1 * Kg^-1]

		# Termo 2: Integral
		# -------------------------------------------

		# inicializa o termo
		termo2 = np.full(self.u.shape, fill_value=np.nan, dtype = np.float64)

		# termos medios
		u_mean = (self.u[:-1] + self.u[1:]) / 2
		T_mean = np.interp(u_mean, self.u, self.T)
		u_vib_mean = np.interp(u_mean, self.u, self.u_vib)
		u_rot_mean = np.interp(u_mean, self.u, self.u_rot)
		u_cont_mean = np.interp(u_mean, self.u, self.u_cont)
		p_mean = np.interp(u_mean, self.u, self.p)
		e_mean = np.interp(u_mean, self.u, self.e)

		# d(Sigma * T(u')^4)/du'
		# dsigma_t_4_du = calc.first_derivative(calc.stefan_boltzmann(self.T), self.u)
		dsigma_t_4_du = np.diff(calc.stefan_boltzmann(self.T)) / du

		# loop para cada nível da sondagem
		for i in range(termo2.shape[0] - 1): # ultimo nao incluso
			
			# Função a ser integrada
			dEfdu = np.full(self.nlevs - 1, fill_value = np.nan)

			# loop para  a integral, da superfície até o topo da sondagem
			for j in range(self.nlevs - 1):
				# Termos medios
				mean_props = dict(
					T = T_mean[j],
					u_rot = u_rot_mean[j],
					u_cont = u_cont_mean[j],
					p = p_mean[j],
					e = e_mean[j]
				)

				# Emissivade até este nivel
				E1 = self._emissivity(
					T_mean[j],
					n1 = j,
					n2 = i + 1,
					u_vib = u_vib_mean[j],
					band = band,
					mean_props = mean_props
					)

				E0 = self._emissivity(
					T_mean[j],
					n1 = j,
					n2 = i,
					u_vib = u_vib_mean[j],
					band = band,
					mean_props = mean_props
					)
				
				dEfdu[j] = (E1 - E0) / (self.u[i + 1] - self.u[0])
					
			# Função a ser integrada
			termo2[i] = np.nansum(dEfdu * dsigma_t_4_du * du) / 10 # Conserta a unidade

		# Cp do ar umido  [J * Kg^-1 * K^-1]
		Cpm = self.Cp * (1 + 0.9 * self.q)

		# Taxa de resfriamento
		cooling_rate = -1 * (termo1 + termo2) * self.q / Cpm # [K/s]

		# converte a unidade para [K / day]
		cooling_rate = cooling_rate * 3600 * 24

		return cooling_rate
	

	def _clear_sky(self, band : str  = 'all'):
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

		# Inicializa a integral [J * m^-2 * s^-1]
		upward_flux = np.full(self.nlevs, fill_value=np.nan)
		downward_flux = np.full(self.nlevs, fill_value = np.nan)

		# loop pra cada nível
		for i in range(self.nlevs): # ultimo nao incluso
			
			# Fluxo ascendente
			upward_flux[i] = self._upward_flux(i, band)
			
			# Fluxo descendente
			downward_flux[i] = self._downward_flux(i, band)

		# Cp do ar umido [J * Kg^-1 * K^-1]
		Cpm = self.Cp * (1 + 0.9 * self.q)

		# Taxa de resfriamento
		dfluxdu = calc.first_derivative(upward_flux - downward_flux, self.u * 10)
		cooling_rate = - 1 * self.q / Cpm * dfluxdu # [K/s]

		# converte a unidade para [K / day]
		cooling_rate = cooling_rate * 3600 * 24

		return cooling_rate
	

	def _upward_flux(self, n, band):

		# Stefan boltman na superficie [J * m^-2 * s^-1]
		surf_flux = calc.stefan_boltzmann(self.T[0])

		# Termo da integral
		integrated_flux = 0

		# loop para a integral, da superfície até o nivel u em questao
		for i in range(n):
			
			# MÉTODO 1
			# ---------
			# # Emissividade
			# Ef = self._emissivity(
			# 	n1 = i,
			# 	n2 = n,
			# 	band = band,
			# 	mean = True
			# 	)

			# # # Derivada de sigma_t_4_du
			# dsigmaT4 = (calc.stefan_boltzmann(self.T[i + 1]) - calc.stefan_boltzmann(self.T[i]))
			
			# # Derivada simples (u está em ordem crescente)
			# integrated_flux += Ef * dsigmaT4

			# MÉTODO 2
			# ---------
			Ef1 = self._emissivity(
				n1 = i + 1,
				n2 = n,
				band = band
			)
			Ef0 = self._emissivity(
				n1 = i,
				n2 = n,
				band = band
			)

			# Funcao
			f1 = Ef1 * self.dSigmaT4du[i + 1]
			f0 = Ef0 * self.dSigmaT4du[i]

			# Valor medio da funcao
			f_ = (f1 + f0) / 2

			# adiciona o fluxo desta camada
			integrated_flux += f_ * (self.u[i + 1] - self.u[i])

		return integrated_flux + surf_flux
	

	def _downward_flux(self, n, band):

		# Stefan boltman no topo [J * m^-2 * s^-1]
		topo = self.nlevs - 1
		top_emissivity = self._emissivity(
			n1 = topo,
			n2 = n,
			band = band
		)
		top_flux = top_emissivity * calc.stefan_boltzmann(self.T[topo])

		# Termo da integral
		integrated_flux = 0

		# loop para a integral, do topo até o nivel u em questao
		for i in range(topo, n, -1):
			
			# Método 1
			# --------
			# # Emissividade
			# Ef = self._emissivity(
			# 	n1 = i,
			# 	n2 = n,
			# 	band = band,
			# 	mean = True)

			# # Derivada de sigma_t_4_du
			# dsigmaT4 = (calc.stefan_boltzmann(self.T[i - 1]) - calc.stefan_boltzmann(self.T[i]))
			
			# # Derivada simples (u está em ordem crescente)
			# integrated_flux += Ef * dsigmaT4

			# MÉTODO 2
			# ---------
			Ef1 = self._emissivity(
				n1 = i - 1,
				n2 = n,
				band = band
			)
			Ef0 = self._emissivity(
				n1 = i,
				n2 = n,
				band = band
			)

			# Funcao
			f1 = Ef1 * self.dSigmaT4du[i - 1]
			f0 = Ef0 * self.dSigmaT4du[i]

			# Valor medio da funcao
			f_ = (f1 + f0) / 2

			# adiciona o fluxo desta camada
			integrated_flux += f_ * (self.u[i - 1] - self.u[i])

		return integrated_flux + top_flux


	def _emissivity(self, n1, n2, band : str = 'all', mean = False, keep_path =False):
		'''
		Calcula a emissividade broadband, considerando todas as bandas ou apenas
		a banda de absorcao especificada pelo usuario.
		Considera o path length do nivel inicial (base) ate um outro nivel final
		da camada atmosferica.

		Parameters
		----------
		n1 : int
			Indice que representa o nivel de onda a emissao parte
		n2 : int
			Indice que representa o nivel para onde a emissao vai
		band : str
			Especifica a banda de absorcao a ser considerada, default = todas.
			Opcoes disponiveis são:
			- 'all' : todas as bandas
			- 'rot' : apenas a banda do rotacional.
			- 'cont' : apenas a banda do continuum.
			- 'vib' : apenas a banda do vibracional-rotacional.
		mean : bool
			Se considera a emissivade da primeira camada a partir de medias ou nao.

		Returns
		-------
		emissivity : float
			Emissividade broadband [adimensional]
		'''

		# step define se o fluxo é ascendente ou descendente.
		if n2 > n1:
			step = 1
			fatia = slice(n1, n2 + step, step)
		elif n2 == n1:
			return 0
		else: # fluxo decrescente
			step = -1
			fatia = slice(n1, n2 + step - self.nlevs, step)
		
		# variaveis
		Tn = self.T[fatia].copy()
		u_rot = self.u_rot[fatia].copy()
		u_cont = self.u_cont[fatia].copy()
		u_vib = self.u_vib[n1]
		p = self.p[fatia].copy()
		e = self.e[fatia].copy()

		# se media, interpola as variaveis para a camada media
		if mean:
			Tn[0] = (self.T[n1] + self.T[n1 + step]) / 2
			p[0] = (self.p[n1] + self.p[n1 + step]) / 2
			e[0] = (self.e[n1] + self.e[n1 + step]) / 2

			if not keep_path:
				u_rot[0] = (self.u_rot[n1] + self.u_rot[n1 + step]) / 2
				u_cont[0] = (self.u_cont[n1] + self.u_cont[n1 + step]) / 2
				u_vib = (self.u_vib[n1] + self.u_vib[n1 + step]) / 2

		# Caso 1: Todas as bandas (rotational, continuum, vibrational-rotational)
		if band == 'all':
			v1, tau1 = broadband.transmitance_rot(Tn, u_rot)
			v2, tau2 = broadband.transmitance_cont(Tn, u_cont, p, e)
			v3, tau3 = broadband.transmitance_vib(
				np.abs(self.u_vib[n2] - u_vib)
				)

			intervalos = np.concatenate((v1, v2, v3)) # Intervalos de numero de onda
			transmitance = np.concatenate((tau1, tau2, tau3)) # Transmitancia em cada intervalo
		
		# Caso 2:
		elif band == 'rot':
			intervalos, transmitance = broadband.transmitance_rot(Tn, u_rot)

		# Caso 3: 
		elif band == 'cont':
			intervalos, transmitance = broadband.transmitance_cont(Tn, u_cont, p, e)

		# Caso 4:
		elif band == 'vib':
			intervalos, transmitance = broadband.transmitance_vib(
				np.abs(self.u_vib[n2] - u_vib)
				)
		
		else: # Lança um erro
			raise KeyError
		
		# Lei de Stefan-Boltzmann
		steboltz = calc.stefan_boltzmann(self.T[n1]) # J * m^-2 * s^-1

		# transforma em m^-1 e converte para frequencia (1/s)
		c = 3 * 1e8
		f = c * (intervalos * 100)
		fmean = np.mean(f, axis = 1) # [1/s]
		df = np.abs(f[:, 1] - f[:, 0]) # [1/s]

		# Calcula a emissividade de cada intervalo da integral (freq) e soma
		emissivity = np.nansum(
			np.pi * calc.planck(fmean, self.T[n1]) * (1 - transmitance) * df / steboltz
			)
		
		return emissivity