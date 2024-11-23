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

# Define constants and function for AEP calculation
Urp = 12
Prated = 13
T = 365*24

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
#weibull = np.zeros((12, 100))

rot_earth = 7.2921e-5
latitude = 55.3
coriolis_parameter = 2 * rot_earth * np.sin(latitude * np.pi / 180)

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

weibull_nyborg = np.zeros((12, 100))
weibull_korsor = np.zeros((12, 100))
weibull_sprogo = np.zeros((12, 100))

# Avoid division by zero 
U = np.maximum(U, 1e-6)

for i in range(12):
    weibull_nyborg[i] = k_weibull[i] / U * (U / A_weibull_nyborg[i]) ** k_weibull[i] * np.exp(
        -(U / A_weibull_nyborg[i]) ** k_weibull[i])
    weibull_korsor[i] = k_weibull[i] / U * (U / A_weibull_korsor[i]) ** k_weibull[i] * np.exp(
        -(U / A_weibull_korsor[i]) ** k_weibull[i])
    weibull_sprogo[i] = k_weibull[i] / U * (U / A_weibull[i]) ** k_weibull[i] * np.exp(
        -(U / A_weibull[i]) ** k_weibull[i])

# I erased the P_15_25 thing 

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

print(f"Total AEP at Sprogo in MWh: {AEP_sprogo[0]:.3f}")
print(f"Total AEP at Nyborg in MWh: {AEP_nyborg[0]:.3f}")
print(f"Total AEP at Korsor in MWh: {AEP_korsor[0]:.3f}")






"Q1a"
k_weibull_reduced = k_weibull * 0.85

# Recalculate Weibull distributions with reduced k parameter
weibull_nyborg_reduced = np.zeros((12, 100))
weibull_korsor_reduced = np.zeros((12, 100))
weibull_sprogo_reduced = np.zeros((12, 100))

for i in range(12):
    weibull_nyborg_reduced[i] = k_weibull_reduced[i] / U * (U / A_weibull_nyborg[i]) ** k_weibull_reduced[i] * np.exp(
        -(U / A_weibull_nyborg[i]) ** k_weibull_reduced[i])
    weibull_korsor_reduced[i] = k_weibull_reduced[i] / U * (U / A_weibull_korsor[i]) ** k_weibull_reduced[i] * np.exp(
        -(U / A_weibull_korsor[i]) ** k_weibull_reduced[i])
    weibull_sprogo_reduced[i] = k_weibull_reduced[i] / U * (U / A_weibull[i]) ** k_weibull_reduced[i] * np.exp(
        -(U / A_weibull[i]) ** k_weibull_reduced[i])

E_sectors_nyborg_reduced = np.zeros((12, 1))
E_sectors_korsor_reduced = np.zeros((12, 1))
E_sectors_sprogo_reduced = np.zeros((12, 1))

for i in range(12):
    E_sectors_nyborg_reduced[i] = power_sector(weibull_nyborg_reduced[i], frequency[i], 1)
    E_sectors_korsor_reduced[i] = power_sector(weibull_korsor_reduced[i], frequency[i], 1)
    E_sectors_sprogo_reduced[i] = power_sector(weibull_sprogo_reduced[i], frequency[i], 1)


AEP_nyborg_reduced = sum(E_sectors_nyborg_reduced)
AEP_korsor_reduced = sum(E_sectors_korsor_reduced)
AEP_sprogo_reduced = sum(E_sectors_sprogo_reduced)

# Print the results
print("AEP with Reduced k:")
print(f"AEP Nyborg (reduced k): {AEP_nyborg_reduced[0]:.2f} MWh")
print(f"AEP Korsor (reduced k): {AEP_korsor_reduced[0]:.2f} MWh")
print(f"AEP Sprogo (reduced k): {AEP_sprogo_reduced[0]:.2f} MWh")


# Calculate the percentage difference
percent_difference_sprog = 100*(AEP_sprogo_reduced - AEP_sprogo) /AEP_sprogo 
percent_difference_ny = 100*(AEP_nyborg_reduced - AEP_nyborg) /AEP_nyborg 
percent_difference_ko = 100*(AEP_korsor_reduced - AEP_korsor) /AEP_korsor 

print(f"Percentage difference in AEP due to 15% reduction in Sprogo: {abs(percent_difference_sprog[0]):.2f}%")
print(f"Percentage difference in AEP due to 15% reduction in Nyborg: {abs(percent_difference_ny[0]):.2f}%")
print(f"Percentage difference in AEP due to 15% reduction in Korsor: {abs(percent_difference_ko[0]):.2f}%")
print("\n")








"Q1b"
# Group timestamp by year
year_counts = sprog['year'].value_counts().sort_index()
# print("Number of Records per Year:")
# print(year_counts)

# The data are really different, so have to select what we find a "complete" year
threshold = year_counts.max() * 0.7
sprog_filtered = sprog[sprog['year'].isin(year_counts[year_counts >= threshold].index)]
avg_wind_speed_per_year = sprog_filtered.groupby('year')['wind_speed'].mean()

# Now to find most windy and weakest, we have to calculate mean wind speed
most_windy_year = avg_wind_speed_per_year.idxmax() 
least_windy_year = avg_wind_speed_per_year.idxmin()

print(f"Most windy year: {most_windy_year}")
print(f"Least windy year: {least_windy_year}")
sprog_most_windy = sprog[sprog['year'] == most_windy_year]
sprog_least_windy = sprog[sprog['year'] == least_windy_year]




# Redo everything of question 2 
frequency_most = np.zeros((12,1))
frequency_least = np.zeros((12,1))

for i in np.arange(1,13,1):
    frequency_most[i-1] = sum((sprog_most_windy['wind_direction'] >= direction_sectors[i-1]) & (sprog_most_windy['wind_direction'] < direction_sectors[i]))/len(sprog_most_windy['wind_direction'])
    frequency_least[i-1] = sum((sprog_least_windy['wind_direction'] >= direction_sectors[i-1]) & (sprog_least_windy['wind_direction'] < direction_sectors[i]))/len(sprog_least_windy['wind_direction'])


A_weibull_most = np.zeros((12, 1))
k_weibull_most = np.zeros((12, 1))
A_weibull_least = np.zeros((12, 1))
k_weibull_least = np.zeros((12, 1))

def weibull_parameters_year(sprog, direction_sector_min, direction_sector_max):
    k_wei, loc, A_wei = s.weibull_min.fit(
        sprog['wind_speed'][(sprog['wind_direction'] < direction_sector_max) & (sprog['wind_direction'] >= direction_sector_min)],
        floc=0)

    return A_wei, k_wei


for i in np.arange(1, 13, 1):
    A_weibull_most[i - 1], k_weibull_most[i - 1] = weibull_parameters_year(sprog_most_windy, direction_sectors[i - 1], direction_sectors[i])
    A_weibull_least[i - 1], k_weibull_least[i - 1] = weibull_parameters_year(sprog_least_windy, direction_sectors[i - 1], direction_sectors[i])


mean_u_sector_most = np.zeros((12, 1))
mean_u_sector_least = np.zeros((12, 1))

for i in range(12):
    mean_u_sector_most[i] = np.mean(
        sprog_most_windy['wind_speed'][(sprog_most_windy['wind_direction'] < direction_sectors[i + 1]) & (sprog_most_windy['wind_direction'] >= direction_sectors[i])])
    mean_u_sector_least[i] = np.mean(
        sprog_least_windy['wind_speed'][(sprog_least_windy['wind_direction'] < direction_sectors[i + 1]) & (sprog_least_windy['wind_direction'] >= direction_sectors[i])])

u_nyborg_most = np.zeros((12, 1))
u_korsor_most = np.zeros((12, 1))
u_nyborg_least = np.zeros((12, 1))
u_korsor_least = np.zeros((12, 1))

u_star_nyborg = np.zeros((12, 1))
u_star_korsor = np.zeros((12, 1))

for i in range(12):
    u_nyborg_most[i], u_star_nyborg[i] = gdl(mean_u_sector_most[i], roughness_nyborg[i])
    u_korsor_most[i], u_star_korsor[i] = gdl(mean_u_sector_most[i], roughness_korsor[i])
    
    u_nyborg_least[i], u_star_nyborg[i] = gdl(mean_u_sector_least[i], roughness_nyborg[i])
    u_korsor_least[i], u_star_korsor[i] = gdl(mean_u_sector_least[i], roughness_korsor[i])


A_weibull_nyborg_most = np.zeros((12, 1))
A_weibull_korsor_most = np.zeros((12, 1))
A_weibull_nyborg_least = np.zeros((12, 1))
A_weibull_korsor_least = np.zeros((12, 1))

for i in range(12):
    A_weibull_nyborg_most[i] = u_nyborg_most[i] / (gamma(1 + 1 / k_weibull[i]))
    A_weibull_korsor_most[i] = u_korsor_most[i] / (gamma(1 + 1 / k_weibull[i]))
    A_weibull_nyborg_least[i] = u_nyborg_least[i] / (gamma(1 + 1 / k_weibull[i]))
    A_weibull_korsor_least[i] = u_korsor_least[i] / (gamma(1 + 1 / k_weibull[i]))

weibull_nyborg_most = np.zeros((12, 100))
weibull_korsor_most = np.zeros((12, 100))
weibull_sprogo_most = np.zeros((12, 100))

weibull_nyborg_least = np.zeros((12, 100))
weibull_korsor_least = np.zeros((12, 100))
weibull_sprogo_least = np.zeros((12, 100))


# Avoid division by zero 
U = np.maximum(U, 1e-6)
for i in range(12):
    weibull_nyborg_most[i] = k_weibull_most[i] / U * (U / A_weibull_nyborg_most[i]) ** k_weibull_most[i] * np.exp(
        -(U / A_weibull_nyborg_most[i]) ** k_weibull_most[i])
    weibull_korsor_most[i] = k_weibull_most[i] / U * (U / A_weibull_korsor_most[i]) ** k_weibull_most[i] * np.exp(
        -(U / A_weibull_korsor_most[i]) ** k_weibull_most[i])
    weibull_sprogo_most[i] = k_weibull_most[i] / U * (U / A_weibull_most[i]) ** k_weibull_most[i] * np.exp(
        -(U / A_weibull_most[i]) ** k_weibull_most[i])
    
    weibull_nyborg_least[i] = k_weibull_least[i] / U * (U / A_weibull_nyborg_least[i]) ** k_weibull_least[i] * np.exp(
        -(U / A_weibull_nyborg_least[i]) ** k_weibull_least[i])
    weibull_korsor_least[i] = k_weibull_least[i] / U * (U / A_weibull_korsor_least[i]) ** k_weibull_least[i] * np.exp(
        -(U / A_weibull_korsor_least[i]) ** k_weibull_least[i])
    weibull_sprogo_least[i] = k_weibull_least[i] / U * (U / A_weibull_least[i]) ** k_weibull_least[i] * np.exp(
        -(U / A_weibull_least[i]) ** k_weibull_least[i])
    


E_sectors_nyborg_most = np.zeros((12, 1))
E_sectors_korsor_most = np.zeros((12, 1))
E_sectors_sprogo_most = np.zeros((12, 1))
E_sectors_nyborg_least = np.zeros((12, 1))
E_sectors_korsor_least = np.zeros((12, 1))
E_sectors_sprogo_least = np.zeros((12, 1))

for i in range(12):
    E_sectors_nyborg_most[i] = power_sector(weibull_nyborg_most[i], frequency_most[i], 1)
    E_sectors_korsor_most[i] = power_sector(weibull_korsor_most[i], frequency_most[i], 1)
    E_sectors_sprogo_most[i] = power_sector(weibull_sprogo_most[i], frequency_most[i], 1)
    
    E_sectors_nyborg_least[i] = power_sector(weibull_nyborg_least[i], frequency_least[i], 1)
    E_sectors_korsor_least[i] = power_sector(weibull_korsor_least[i], frequency_least[i], 1)
    E_sectors_sprogo_least[i] = power_sector(weibull_sprogo_least[i], frequency_least[i], 1)

AEP_nyborg_most = sum(E_sectors_nyborg_most)[0]
AEP_korsor_most = sum(E_sectors_korsor_most)[0]
AEP_sprogo_most = sum(E_sectors_sprogo_most)[0]

AEP_nyborg_least = sum(E_sectors_nyborg_least)[0]
AEP_korsor_least = sum(E_sectors_korsor_least)[0]
AEP_sprogo_least = sum(E_sectors_sprogo_least)[0]

print(f"Total AEP at Sprogo for windiest in MWh: {AEP_sprogo_most:.3f}")
print(f"Total AEP at Nyborg for windiest in MWh: {AEP_nyborg_most:.3f}")
print(f"Total AEP at Korsor for windiest in MWh: {AEP_korsor_most:.3f}")

print(f"Total AEP at Sprogo for weakest in MWh: {AEP_sprogo_least:.3f}")
print(f"Total AEP at Nyborg for weakest in MWh: {AEP_nyborg_least:.3f}")
print(f"Total AEP at Korsor for weakest in MWh: {AEP_korsor_least:.3f}")




"Q1b i)"
# "Mean AEP" are above
diff_most_sprogo = (100*(AEP_sprogo_most- AEP_sprogo)/AEP_sprogo)[0]
diff_least_sprogo = (100*(AEP_sprogo_least - AEP_sprogo)/AEP_sprogo)[0]
diff_most_ny = (100*(AEP_nyborg_most - AEP_nyborg)/AEP_nyborg)[0]
diff_least_ny = (100*(AEP_nyborg_least - AEP_nyborg)/AEP_nyborg)[0]
diff_most_ko = (100*(AEP_korsor_most - AEP_korsor)/AEP_korsor)[0]
diff_least_ko = (100*(AEP_korsor_least - AEP_korsor)/AEP_korsor)[0]

print(f"Percentage diff most Sprogo: {diff_most_sprogo:.3f}")
print(f"Percentage diff least Sprogo: {diff_least_sprogo:.3f}")
print(f"Percentage diff most Nyborg: {diff_most_ny:.3f}")
print(f"Percentage diff least Nyborg: {diff_least_ny:.3f}")
print(f"Percentage diff most Korsor: {diff_most_ko:.3f}")
print(f"Percentage diff least Korsor: {diff_least_ko:.3f}")
print("\n")



"Q1b ii)"
# Calculate standard deviation on wind speeds and explain with AEP results  
avg_wind_speed_sprog = sprog['wind_speed'].mean()
avg_wind_speed_most_windy = sprog[sprog['year'] == most_windy_year]['wind_speed'].mean()
avg_wind_speed_least_windy = sprog[sprog['year'] == least_windy_year]['wind_speed'].mean()

print(f"Average wind speed at Sprogø across all years: {avg_wind_speed_sprog:.2f} m/s")
print(f"Average wind speed in the most windy year ({most_windy_year}): {avg_wind_speed_most_windy:.2f} m/s")
print(f"Average wind speed in the least windy year ({least_windy_year}): {avg_wind_speed_least_windy:.2f} m/s")

# Long-term average wind speed (mean of all data)
std_dev_annual_means = avg_wind_speed_per_year.std()
print(f"Standard deviation of annual mean wind speeds at Sprogø: {std_dev_annual_means:.2f} m/s")





print('End')