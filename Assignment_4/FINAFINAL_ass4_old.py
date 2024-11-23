### IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
import matplotlib as matplot
from scipy import stats
import scipy.stats as s
from scipy.special import gamma

from IPython import get_ipython
#get_ipython().magic('reset -f') # clear all variables
#plt.close("all") # close all figures


### READ DATA
headers = ['timestamp', 'wind_speed', 'wind_direction_67.5m', 'wind_direction_70m']
sprog = pd.read_csv("sprog.tsv", sep='\t', header=None, names=headers)

### DATA CLEANING
sprog = sprog.apply(pd.to_numeric, errors='coerce')

sprog.replace(999, np.nan, inplace=True)
sprog['wind_direction'] = sprog['wind_direction_67.5m'].combine_first(sprog['wind_direction_70m'])
sprog[sprog['wind_direction_67.5m'].isna() & sprog['wind_direction_70m'].notna()]
sprog[sprog['wind_direction_67.5m'].isna()]

sprog.replace(99.99, np.nan, inplace=True)

sprog = sprog.dropna(subset=['wind_speed'])
sprog = sprog.dropna(subset=['wind_direction'])

# For the windiest years and cleaning of small years
sprog['timestamp'] = pd.to_datetime(sprog['timestamp'], format='%Y%m%d%H%M')
sprog['year'] = sprog['timestamp'].dt.year


### PART 1
## 1

data = {
    "sector": [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330],
    "A": [7.750969, 7.525115, 7.761153, 9.236267, 9.603213, 8.461672, 9.407033, 9.931700, 9.951515, 10.194087, 9.630792, 7.896237],
    "k": [1.900057, 1.894834, 2.046636, 2.311559, 2.316891, 2.064537, 2.157974, 2.468474, 2.553114, 2.563006, 2.200351, 1.895504]
}

# Create DataFrame
Weib_parm = pd.DataFrame(data)
Weib_parm_dic = {row['sector']: {'A': row['A'], 'k': row['k']} for _, row in Weib_parm.iterrows()}

# Direction segments
bins = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360])
labels = np.arange(0, 360, 30)

sprog['sector'] = pd.cut(sprog['wind_direction'], bins=bins, labels=labels, right=False)
sprog.loc[(sprog['wind_direction'] >= 345) | (sprog['wind_direction'] < 15), 'sector'] = 0

sector_dfs = {sector: sprog[sprog['sector'] == sector] for sector in sprog['sector'].unique()}
sector_dfs.pop(np.nan, None)

# Define constants and function for AEP calculation
Urp = 12
Prated = 13
T = 365*24


## TASK 1

direction_sectors = np.arange(0,390,30)
frequency = np.zeros((12,1))

for i in np.arange(1,13,1):

    frequency[i-1] = sum((sprog['wind_direction'] >= direction_sectors[i-1]) & (sprog['wind_direction'] < direction_sectors[i]))/len(sprog['wind_direction'])

z_measurement = 70
u_mean = np.mean(sprog['wind_speed'])
vonKarman = 0.4
z_0_water = 0.0002
z_0_land = 0.02
z_hub = 120

A_weibull = np.zeros((12, 1))
k_weibull = np.zeros((12, 1))


def weibull_parameters(direction_sector_min, direction_sector_max):
    k_wei, loc, A_wei = s.weibull_min.fit(
        sprog['wind_speed'][(sprog['wind_direction'] < direction_sector_max) & (sprog['wind_direction'] >= direction_sector_min)],
        floc=0)

    return A_wei, k_wei


for i in np.arange(1, 13, 1):
    A_weibull[i - 1], k_weibull[i - 1] = weibull_parameters(direction_sectors[i - 1], direction_sectors[i])

U = np.linspace(0, 30, 100)
weibull = np.zeros((12, 100))

# Obtaining coriolis parameter from latitude

rot_earth = 7.2921e-5
latitude = 55.3

coriolis_parameter = 2 * rot_earth * np.sin(latitude * np.pi / 180)

# U-components using A and B from Mark's notes

A = 1.8
B = 4.5

# GDL function

mean_u_sector_sprogo = np.zeros((12, 1))

for i in range(12):
    mean_u_sector_sprogo[i] = np.mean(
        sprog['wind_speed'][(sprog['wind_direction'] < direction_sectors[i + 1]) & (sprog['wind_direction'] >= direction_sectors[i])])


def gdl(mean_u, z_0):
    u_star = mean_u * vonKarman / (np.log(z_measurement / z_0_water))

    u_g = u_star / vonKarman * (np.log((u_star / coriolis_parameter) / z_0_water) - A)

    v_g = - B * u_star / vonKarman

    G = np.sqrt(u_g ** 2 + v_g ** 2)

    u_star_guess = 0.0001

    error = 1

    while error > 0.001:
        error = G - u_star_guess / vonKarman * np.sqrt(
            (np.log(u_star_guess / (coriolis_parameter * z_0)) - A) ** 2 + B ** 2)

        u_star_guess = u_star_guess + 0.0001

    mean_u_new = u_star_guess / vonKarman * np.log(z_hub / z_0)

    return mean_u_new, u_star_guess


roughness_nyborg = np.zeros((12, 1))

roughness_korsor = np.zeros((12, 1))

roughness_nyborg[direction_sectors[0:12] <= 180] = z_0_water
roughness_nyborg[direction_sectors[0:12] > 180] = z_0_land

roughness_korsor[direction_sectors[0:12] > 180] = z_0_water
roughness_korsor[direction_sectors[0:12] <= 180] = z_0_land

u_nyborg = np.zeros((12, 1))
u_korsor = np.zeros((12, 1))
u_star_nyborg = np.zeros((12, 1))
u_star_korsor = np.zeros((12, 1))

for i in range(12):
    u_nyborg[i], u_star_nyborg[i] = gdl(mean_u_sector_sprogo[i], roughness_nyborg[i])

    u_korsor[i], u_star_korsor[i] = gdl(mean_u_sector_sprogo[i], roughness_korsor[i])

A_weibull_nyborg = np.zeros((12, 1))
A_weibull_korsor = np.zeros((12, 1))

for i in range(12):
    A_weibull_nyborg[i] = u_nyborg[i] / (gamma(1 + 1 / k_weibull[i]))
    A_weibull_korsor[i] = u_korsor[i] / (gamma(1 + 1 / k_weibull[i]))

T = 8760

weibull_nyborg = np.zeros((12, 100))
weibull_korsor = np.zeros((12, 100))
weibull_sprogo = np.zeros((12, 100))

for i in range(12):
    weibull_nyborg[i] = k_weibull[i] / U * (U / A_weibull_nyborg[i]) ** k_weibull[i] * np.exp(
        -(U / A_weibull_nyborg[i]) ** k_weibull[i])
    weibull_korsor[i] = k_weibull[i] / U * (U / A_weibull_korsor[i]) ** k_weibull[i] * np.exp(
        -(U / A_weibull_korsor[i]) ** k_weibull[i])
    weibull_sprogo[i] = k_weibull[i] / U * (U / A_weibull[i]) ** k_weibull[i] * np.exp(
        -(U / A_weibull[i]) ** k_weibull[i])

P_15_25_nyborg = np.zeros((12, 1))
P_15_25_korsor = np.zeros((12, 1))
P_15_25_sprogo = np.zeros((12, 1))

for i in range(12):
    P_15_25_nyborg[i] = sum(weibull_nyborg[i, 50:83]) * 1 / 3
    P_15_25_korsor[i] = sum(weibull_korsor[i, 50:83]) * 1 / 3
    P_15_25_sprogo[i] = sum(weibull_sprogo[i, 50:83]) * 1 / 3
print('here')

P_15_25_nyborg = np.mean(P_15_25_nyborg) * 100
P_15_25_korsor = np.mean(P_15_25_korsor) * 100
P_15_25_sprogo = np.mean(P_15_25_sprogo) * 100

U_rp = 12
P_rated = 13


def power_sector(probability, fre, wrf):
    E = 0

    for i in range(len(U) - 1):

        if U[i + 1] <= U_rp:

            E = E + (P_rated * (U[i + 1] / U_rp) ** 3) * probability[i + 1] * fre * T * 0.3

        elif U_rp < U[i + 1] < 25 * wrf:

            E = E + P_rated * probability[i + 1] * fre * T * 0.3

    return E


E_sectors_nyborg = np.zeros((12, 1))
E_sectors_korsor = np.zeros((12, 1))
E_sectors_sprogo = np.zeros((12, 1))

for i in range(12):
    E_sectors_nyborg[i] = power_sector(weibull_nyborg[i], frequency[i], 1)

    E_sectors_korsor[i] = power_sector(weibull_korsor[i], frequency[i], 1)

    E_sectors_sprogo[i] = power_sector(weibull_sprogo[i], frequency[i], 1)

AEP_nyborg = sum(E_sectors_nyborg)
AEP_korsor = sum(E_sectors_korsor)
AEP_sprogo = sum(E_sectors_sprogo)

print('End')