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
import Scripts.broadband as broadband
import Scripts.calc as calc


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

	def __init__(self, T, u, q, p, Qv, z, nlevels = None):
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
		z : Array[float]
			Altitude de cada nivel da sondagem [metros]
		nlevels : int [opcional]
			Interpola a sondagem para "nlevels" niveis.
		'''

		# Define as variáveis privadas desta classe
		self.T = T
		self.u = u if u[1] - u[0] > 0 else u[0] - u # Garante que "u" é crescente
		self.q = q
		self.p = p
		self.Qv = Qv
		self.z = z

		# Interpola para n niveis da sondagem, se especificado pelo usuario
		if nlevels is not None:
			new_z = np.linspace(self.z[0], self.z[-1], nlevels)
			self.T = np.interp(new_z, self.z, self.T)
			self.q = np.interp(new_z, self.z, self.q)
			self.p = np.interp(new_z, self.z, self.p)
			self.Qv = np.interp(new_z, self.z, self.Qv)
			self.u = np.interp(new_z, self.z, self.u)
			self.z = new_z

		# Numero de niveis
		self.nlevs = self.u.shape[0]

		# Calcula parâmetros que serão utilizados, baseado nas variaveis acima
		# ---------------------------------------------------------------------
		# Pressão parcial do vapor na parcela de ar [Pa]
		self.e = self.Qv * self.Rv * self.T  # Lei do gás ideal 

		# Path length (u) corrigido para cada banda de absorcao
		self.u_rot = broadband.path_length_rot(self.T, self.p, self.u)
		self.u_cont = broadband.path_length_cont(self.T, self.u, self.e)
		self.u_vib = broadband.path_length_vib(self.u, self.p)

		# Derivadas importantes
		# ---------------------------------------------------------------------
		self.dSigmaT4du = calc.first_derivative(
			calc.stefan_boltzmann(self.T), self.u)

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
		# ---------------------
		dfluxdu = calc.first_derivative(upward_flux - downward_flux, self.u * 10)
		cooling_rate = - 1 * self.q / Cpm * dfluxdu # [K/s]

		# converte a unidade para [K / day]
		cooling_rate = cooling_rate * 3600 * 24

		return cooling_rate
	
	def cloudy_atmosphere(self, band : str, Ect : float, Ecb : float, Tc : float, Rc : float, Zct : int, Zcb : int):
		'''
		Taxa de resfriamento considerando um céu com nuvem.

		Parameters
		----------
		band : str
			Especifica a banda de absorcao a ser considerada, default = todas.
			Opcoes disponiveis são:
			- 'all' : todas as bandas
			- 'rot' : apenas a banda do rotacional.
			- 'cont' : apenas a banda do continuum.
			- 'vib' : apenas a banda do vibracional-rotacional.

		Ect : float
			Emissividade do topo da nuvem, no intervalo [0, 1].

		Ecb : float
			Emissividade da base da nuvem, no intervalo [0, 1].

		Tc : float
			Transmissividade da nuvem, no intervalo [0, 1].

		Rc : float
			refletividade da nuvem, no intervalo [0, 1].

		Ncb : int
			Nível da base da nuvem na sondagem [m].
		
		Nct : int
			Nível do topo da nuvem na sondagem [m].

		Returns
		-------
		cooling_rate : array[float]
			Taxa de resfriamento [Kelvin / dia]
		'''
		# variaveis privadas
		self._Ect = Ect
		self._Ecb = Ecb
		self._Tc = Tc
		self._Rc = Rc

		# Definindo o indice da base e topo da nuvem 
		self._Ncb = np.count_nonzero(self.z < Zcb)
		self._Nct = np.count_nonzero(self.z < Zct) - 1

		# Inicializa a integral [J * m^-2 * s^-1]
		upward_flux = np.full(self.nlevs, fill_value=np.nan)
		downward_flux = np.full(self.nlevs, fill_value = np.nan)

		# loop pra cada nível
		for i in range(self.nlevs): # ultimo nao incluso
			# Verifica se este nível é a nuvem
			if i >= self._Ncb and i <= self._Nct:
				continue

			# Fluxo ascendente, checa se este nível está acima da nuvem
			if i > self._Nct:
				upward_flux[i] = self._cloud_upward_flux(i, band)
			else:
				upward_flux[i] = self._upward_flux(i, band)

			# Fluxo descendente, checa se este nível está abaixo da nuvem
			if i < self._Ncb :
				downward_flux[i] = self._cloud_downward_flux(i, band)
			else:

				downward_flux[i] = self._downward_flux(i, band)

		# Cp do ar umido [J * Kg^-1 * K^-1]
		Cpm = self.Cp * (1 + 0.9 * self.q)

		# Taxa de resfriamento
		# ---------------------
		diff = upward_flux - downward_flux
		u_ = self.u * 10
		dfluxdu = np.full(upward_flux.shape, fill_value= np.nan)
		
		# Niveis abaixo da nuvem
		dfluxdu[:self._Ncb] = calc.first_derivative(diff[:self._Ncb], u_[:self._Ncb])

		# Niveis acima da nuvem
		dfluxdu[self._Nct + 1:] = calc.first_derivative(diff[self._Nct + 1:], u_[self._Nct + 1:])

		# taxa
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

	def _cloud_upward_flux(self, n, band):
		# INTEGRAL
		# Fluxo que desce pela emissão de niveis acima da nuvem, que reflete no topo da nuvem e volta para o nivel n
		# ----------------------------------------------------------------------------
		fluxo_1 = 0 # inicializa como zero e vai somando

		# loop da integral, do nível TOPO até Nct
		topo = self.nlevs - 1
		for i in range(topo, self._Ncb, -1):

			# MÉTODO 1
			# ---------
			irradiancia = calc.stefan_boltzmann(
				(self.T[i - 1] + self.T[i]) / 2
			)
			
			# transmitancia do nivel i até bater na nuvem e refletir até o nivel N
			t1 =  self._transmitance_leap([i, self._Nct, self._Nct, n], band)

			# transmitancia do nivel i + 1 até bater na nuvem e refletir até o nivel N
			t2 =  self._transmitance_leap([i - 1, self._Nct, self._Nct, n], band)
			
			# adiciona o fluxo desta camada
			fluxo_1 += irradiancia * (t2 - t1)

		# Fluxo da superficie, que passa pela nuvem e chega em N
		# na base da nuvem e volta pra n
		# ---------------------------------------------------------------------------------------
		fluxo_2 = calc.stefan_boltzmann(self.T[0]) * self._transmitance_leap([0, self._Ncb, self._Nct, n], band)
		
		# INTEGRAL
		# Fluxo pela emissão dos níveis abaixo da nuvem, transmitido através da nuvem e chega em n
		# ---------------------------------------------------------------------------
		fluxo_3 = 0 # inicializa como zero e vai somando
		
		# loop da integral, da sup até a base da nuvem
		for i in range(0, self._Ncb):

			# MÉTODO 1
			# ---------
			irradiancia = calc.stefan_boltzmann(
				(self.T[i + 1] + self.T[i]) / 2
			)
			
			# transmitancia do nivel i até passar pela nuvem e chegar ao nivel N
			t1 =  self._transmitance_leap([i, self._Ncb, self._Nct, n], band)

			# ttransmitancia do nivel i + 1 até passar pela nuvem e chegar ao nivel N
			t2 =  self._transmitance_leap([i + 1, self._Ncb, self._Nct, n], band)
			
			# adiciona o fluxo desta camada
			fluxo_3 += irradiancia * (t2 - t1)

		# Fluxo devido a emissão do topo da nuvem
		# ---------------------------------------------------------------------------
		fluxo_4 = self._Ect * (1 - self._emissivity(self._Nct, n, band)) * calc.stefan_boltzmann(self.T[self._Nct])

		# INTEGRAL
		# Fluxo ascendente das camadas entre o topo da nuvem e o nivel n
		# ----------------------------------------------------------------
		fluxo_5 = 0 # inicializa como zero e vai somando

		# loop da integral, da topo da nuvem até n
		for i in range(self._Nct, n):

			# MÉTODO 1
			# ---------
			irradiancia = calc.stefan_boltzmann(
				(self.T[i + 1] + self.T[i]) / 2
			)
			
			# transmitancia do nivel i até passar pela nuvem e chegar ao nivel N
			t1 =  1 - self._emissivity(i, n, band)

			# ttransmitancia do nivel i - 1 até passar pela nuvem e chegar ao nivel N
			t2 =  1 - self._emissivity(i + 1, n, band)
			
			# adiciona o fluxo desta camada
			fluxo_5 += irradiancia * (t2 - t1)

		# Resultado final
		upward_flux = self._Rc * fluxo_1 + self._Tc * (fluxo_2 + fluxo_3) + fluxo_4 + fluxo_5

		return upward_flux

	def _cloud_downward_flux(self, n, band):
		# Fluxo da superfície que é refletida na base da nuvem e volta para o nível i
		# ----------------------------------------------------------------------------
		fluxo_1 = calc.stefan_boltzmann(self.T[0]) * self._transmitance_leap([0, self._Ncb, self._Ncb, n], band)

		# INTEGRAL
		# Fluxo pela emissão dos demais níveis abaixo da nuvem, que é refletida
		# na base da nuvem e volta pra n
		# ---------------------------------------------------------------------------------------
		fluxo_2 = 0 # inicializa como zero e vai somando
		
		# loop da integral, do nível 0 até Ncb
		for i in range(self._Ncb):

			# MÉTODO 1
			# ---------
			irradiancia = calc.stefan_boltzmann(
				(self.T[i + 1] + self.T[i]) / 2
			)
			
			# transmitancia do nivel i até bater na nuvem e refletir até o nivel N
			t1 =  self._transmitance_leap([i, self._Ncb, self._Ncb, n], band)

			# transmitancia do nivel i + 1 até bater na nuvem e refletir até o nivel N
			t2 =  self._transmitance_leap([i + 1, self._Ncb, self._Ncb, n], band)
			
			# adiciona o fluxo desta camada
			fluxo_2 += irradiancia * (t2 - t1)

		# INTEGRAL
		# Fluxo pela emissão dos níveis acima da nuvem, transmitido através da nuvem
		# ---------------------------------------------------------------------------
		fluxo_3 = 0 # inicializa como zero e vai somando
		
		# loop da integral, do topo até Nct
		topo = self.nlevs - 1
		for i in range(topo, self._Nct, -1):

			# MÉTODO 1
			# ---------
			irradiancia = calc.stefan_boltzmann(
				(self.T[i - 1] + self.T[i]) / 2
			)
			
			# transmitancia do nivel i até passar pela nuvem e chegar ao nivel N
			t1 =  self._transmitance_leap([i, self._Nct, self._Ncb, n], band)

			# ttransmitancia do nivel i - 1 até passar pela nuvem e chegar ao nivel N
			t2 =  self._transmitance_leap([i - 1, self._Nct, self._Ncb, n], band)
			
			# adiciona o fluxo desta camada
			fluxo_3 += irradiancia * (t2 - t1)

		# Fluxo devido a emissão da base da nuvem para baixo.
		# ---------------------------------------------------------------------------
		fluxo_4 = self._Ecb * (1 - self._emissivity(self._Ncb, n, band)) * calc.stefan_boltzmann(self.T[self._Ncb])

		# INTEGRAL
		# Fluxo descendente das camadas entre a base da nuvem e o nivel n
		# ----------------------------------------------------------------
		fluxo_5 = 0 # inicializa como zero e vai somando

		# loop da integral, da base até n
		for i in range(self._Ncb, n, -1):

			# MÉTODO 1
			# ---------
			irradiancia = calc.stefan_boltzmann(
				(self.T[i - 1] + self.T[i]) / 2
			)
			
			# transmitancia do nivel i até passar pela nuvem e chegar ao nivel N
			t1 =  1 - self._emissivity(i, n, band)

			# ttransmitancia do nivel i - 1 até passar pela nuvem e chegar ao nivel N
			t2 =  1 - self._emissivity(i - 1, n, band)
			
			# adiciona o fluxo desta camada
			fluxo_5 += irradiancia * (t2 - t1)

		# Resultado final
		downward_flux = self._Rc * (fluxo_1 + fluxo_2) + self._Tc * fluxo_3 + fluxo_4 + fluxo_5

		return downward_flux

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

		# Funcao de transmissão difusa
		du_vib = self.u_vib[n2] - u_vib
		intervalos, transmitance = self._diffuse_transmission_function(
			band, Tn, p, e, u_rot, u_cont, du_vib
		)

		# Calcula a emissividade em todo o espectro
		result = self._broadband_emissivity(
			self.T[n1], intervalos, transmitance
		)
		
		return result
		
	def _transmitance_leap(self, levels, band : str = 'all'):
		'''
		Calcula a transmitancia broadband, considerando todas as bandas ou apenas
		a banda de absorcao especificada pelo usuario.
		Considera o path length do nivel inicial (base) ate um outro nivel final
		da camada atmosferica.

		Parameters
		----------
		levels : list[int]
			Lista de indices que representam os niveis por onde a emissao passa
		band : str
			Especifica a banda de absorcao a ser considerada, default = todas.
			Opcoes disponiveis são:
			- 'all' : todas as bandas
			- 'rot' : apenas a banda do rotacional.
			- 'cont' : apenas a banda do continuum.
			- 'vib' : apenas a banda do vibracional-rotacional.

		Returns
		-------
		transmitance: float
			Transmitancia broadband [adimensional]
		'''

		transmitances = []
		for i in range(2):
			# step define se o fluxo é ascendente ou descendente.
			if levels[i*2 + 1] > levels[i*2]:
				step = 1
				fatia = slice(levels[i*2], levels[i*2 + 1], step)
			else: # fluxo decrescente
				step = -1
				fatia = slice(levels[i*2], levels[i*2 + 1] - self.nlevs, step)

			# variaveis
			Tn = self.T[fatia].copy()
			u_rot = self.u_rot[fatia].copy()
			u_cont = self.u_cont[fatia].copy()
			p = self.p[fatia].copy()
			e = self.e[fatia].copy()
			du_vib = self.u_vib[i*2 + 1] - self.u_vib[i*2]

			# Funcao de transmissão difusa
			intervalos, tr = self._diffuse_transmission_function(
				band, Tn, p, e, u_rot, u_cont, du_vib
			)

			# Coloca na lista
			transmitances.append(tr)

		# junta as transmitancias da primeira e segunda trajetorias
		transmitance = transmitances[1] * transmitances[0]

		# Calcula a transmitacia (1 - emissividade) em todo o espectro
		result = 1 - self._broadband_emissivity(
			self.T[levels[0]], intervalos, transmitance
		)
		
		return result
	
	def _diffuse_transmission_function(self, band, T, p, e, u_rot, u_cont, du_vib):
		'''
		Calcula a transmitancia difusa para cada banda do espectro, dado o caminho optico
		e as propriedades atmosférias em cada parte do caminho.

		Parameters
		----------
		band : str
			Especifica a banda de absorcao a ser considerada, default = todas.
			Opcoes disponiveis são:
			- 'all' : todas as bandas
			- 'rot' : apenas a banda do rotacional.
			- 'cont' : apenas a banda do continuum.
			- 'vib' : apenas a banda do vibracional-rotacional.
		T : Array[float]
			A temperatura do ar [K] em cada nivel u
		p : Array[float]
			Pressão atmosférica [hPa] em cada nivel u
		e : Array[float]
			Pressão do vapor d'água [hPa] em cada nivel u
		u_rot : Array[float]
			Path Length [g / cm²] corrigido no rotacional
		u_cont : Array[float]
			Path Length [g / cm²] corrigido no continuum
		du_vib : float
			Path Length [g / cm²] total corrigido no vibracional-rotacional

		Returns
		-------
		intervalos: array[float]
			Intervalos de numero de onda, ou frequencia, para cada banda do espectro
		transmitance: array[float]
			Transmitancia difusa [adimensional] para cada banda do espectro
		'''

		# Caso 1: Todas as bandas (rotational, continuum, vibrational-rotational)
		if band == 'all':
			v1, tau1 = broadband.transmitance_rot(T, u_rot)
			v2, tau2 = broadband.transmitance_cont(T, u_cont, p, e)
			v3, tau3 = broadband.transmitance_vib(np.abs(du_vib))

			intervalos = np.concatenate((v1, v2, v3)) # Intervalos de numero de onda
			transmitance = np.concatenate((tau1, tau2, tau3)) # Transmitancia em cada intervalo
		
		# Caso 2:
		elif band == 'rot':
			intervalos, transmitance = broadband.transmitance_rot(T, u_rot)

		# Caso 3: 
		elif band == 'cont':
			intervalos, transmitance = broadband.transmitance_cont(T, u_cont, p, e)

		# Caso 4:
		elif band == 'vib':
			intervalos, transmitance = broadband.transmitance_vib(du_vib)
		
		else: # Lança um erro
			raise KeyError
		
		return intervalos, transmitance
	
	def _broadband_emissivity(self, T, intervalos, transmitance):
		'''
		Calcula a transmitancia difusa para cada banda do espectro, dado o caminho optico
		e as propriedades atmosférias em cada parte do caminho.

		Parameters
		----------
		T : Array[float]
			A temperatura do ar [K] da fonte emissora
		intervalos: array[float]
			Intervalos de numero de onda, ou frequencia, para cada banda do espectro
		transmitance: array[float]
			Transmitancia difusa [adimensional] para cada banda do espectro

		Returns
		-------
		emissivity : float
			Emissividade broadband [adimensional]
		'''
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