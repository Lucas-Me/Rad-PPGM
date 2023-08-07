# =============================
# script para testes

# Made by: Lucas da Silva Menezes
# 31/07/2023
# =============================


# IMPORTS
import pandas as pd
import matplotlib.pyplot as plt

# import locais
from calc import *
import cooling_rate

# =============================

def wyoming_read(file_path):

	# NOME DE CADA COLUNA NA SONDAGEM | uniddde
	colunas = [
		'pres', # hPa
		'hght', # m
		'temp', # °C
		'dwpt', # °C
		'relh', # %
		'mixr', # g / kg
		'drct', # ° [0, 360]
		'sknt', # Knot
		'thta', # K
		'thte', # K
		'thtv' # K
	]

	# lendo o arquivo
	df = pd.read_csv(
		file_path,
		sep = '\s+',
		names = colunas
	)

	# convertendo para as unidades que serao usadas
	df['temp'] = df.temp + 273.15 # °C para K
	df['dwpt'] = df.dwpt + 273.15 # °C para K

	# Converte a razao de mistura pra kg / kg
	df['mixr'] = df['mixr'] * 1e-3

	# Filtra para z < 15 km e as pega as colunas necessarias
	df = df.loc[df['hght'] < 15000][colunas[:-5]]

	# Calcula a densidade do vapor
	df['vapor_density'] = density_water_vapor(
		T = df['temp'],
		Td  = df['dwpt'],
		p = df['pres'],
		w = df['mixr'] # kg / kg
	) # em Kg / m³

	# Calcula o path length [Kg / m²]
	df['u'] = path_length(df['vapor_density'].values, df['hght'].values)

	# converte para g/cm²
	df['u'] = df['u'] * 1e-1

	# filtra e retorna
	return df


def read_test(path_name):
	df = pd.read_csv(path_name, header = 0)

	# Calcula a razao de mistura [g/kg]
	df['mixr'] = df['water_dens'] / df['air_dens'] * 1e3

	# Calcula o path length [Kg / m²]
	df['u'] = path_length(df['water_dens'].values * 1e-3, df['hght'].values)

	# converte para g/cm²
	df['u'] = df['u'] * 1e-1

	# filtra e retorna
	return df

def figure(df):
	fig, ax  = plt.subplots(figsize = (5, 9))

	ax.plot(df['cr_nc_all'], df['hght'] * 1e-3, linestyle = 'solid', color = 'k', label = 'total')
	ax.plot(df['cr_nc_rot'], df['hght'] * 1e-3, linestyle =  'dashed', label = 'H20 Rotational')
	ax.plot(df['cr_nc_cont'], df['hght'] * 1e-3, linestyle = 'dashdot', label = 'H20 Continuum')
	ax.plot(df['cr_nc_vib'], df['hght'] * 1e-3, linestyle = 'dotted', label = 'H20 Vibrational')

	# Eixo Y
	ax.set_ylim(0, 15)
	ax.set_yticks([0, 5, 10, 15])

	# Eixo X
	left, right = ax.get_xlim()
	ax.set_xticks(np.arange(-20, 20, 1))
	ax.set_xlim(left, right)

	# Textos
	ax.set_ylabel("Altitude (Km)")
	ax.set_xlabel("Cooling rate (K / day)")
	
	# LEgenda e salva
	plt.legend()
	name = r'D:\Lucas\Mestrado\Disciplinas\Radiacao Solar Terrestre\Rad-PPGM\resultado.png'
	fig.savefig(name, bbox_inches = 'tight', dpi = 200)
	plt.close()


if __name__ == '__main__':
	# df = wyoming_read(r"Dados/sbgl_00z_06ago23.txt")
	df = read_test(r"D:\Lucas\Mestrado\Disciplinas\Radiacao Solar Terrestre\Rad-PPGM\Dados\perfil_vinicius.txt")

	# Cooling rate (sem nuvens)
	df['cr_nc_rot'] = cooling_rate.no_clouds(
		T = df['temp'].values,
		u = df['u'].values,
		q = df['mixr'].values * 1e-3,
		p = df['pres'].values,
		band = 'rotational' # Todas as bandas
	)

	df['cr_nc_cont'] = cooling_rate.no_clouds(
		T = df['temp'].values,
		u = df['u'].values,
		q = df['mixr'].values * 1e-3,
		p = df['pres'].values,
		band = 'continuum' # Todas as bandas
	)

	df['cr_nc_vib'] = cooling_rate.no_clouds(
		T = df['temp'].values,
		u = df['u'].values,
		q = df['mixr'].values * 1e-3,
		p = df['pres'].values,
		band = 'vibrational' # Todas as bandas
	)

	df['cr_nc_all'] = cooling_rate.no_clouds(
		T = df['temp'].values,
		u = df['u'].values,
		q = df['mixr'].values * 1e-3,
		p = df['pres'].values,
		band = 'all' # Todas as bandas
	)

	print(df)
	df.to_csv(r'D:\Lucas\Mestrado\Disciplinas\Radiacao Solar Terrestre\Rad-PPGM\resultado.csv', index = False)
	figure(df)