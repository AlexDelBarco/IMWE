### IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp 

from IPython import get_ipython
get_ipython().magic('reset -f') # clear all variables
plt.close("all") # close all figures



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

#Remove NaN rows
sprog = sprog.dropna(subset=['wind_speed'])
sprog = sprog.dropna(subset=['wind_direction'])

sprog['timestamp'] = pd.to_datetime(sprog['timestamp'], format='%Y%m%d%H%M')
sprog['year'] = sprog['timestamp'].dt.year



################ PART 2 : EXTREME WIND SPEED ###################
mean_WS = sprog['wind_speed'].mean()
std_WS = sprog['wind_speed'].std()
var_WS = sprog['wind_speed'].var()

print(f"Mean Wind Speed: {mean_WS:.2f} m/s")
print(f"Standard Deviation: {std_WS:.2f} m/s")
print(f"Variance: {var_WS:.2f} m/s²\n")


# Define constants
T = 365*24       # hours in a year
z0_w = 0.02e-2   #in m, roughness over water 
z0_l = 3e-2
z_hub = 112
kappa = 0.4

gamma = 0.5772 # Euler-Mascheroni constant
T = 50
T_0 = 1 # Reference time, 1 year since we have a couple a data this should be fine


############# PWM METHOD ###############
windspeed = sprog['wind_speed']
N = len(windspeed)
sorted_speeds = np.sort(windspeed)

# Calculate the PWM moments
b0 = mean_WS          # b0 is the mean of the sorted data
b1 = np.sum((np.arange(1, N+1) - 1) * sorted_speeds) / (N * (N-1))

alpha = 2*b1- b0*np.log(2)
beta = b0 - gamma * alpha
U50 = alpha * np.log(T/T_0) + beta

print(f"PWM Method:")
print(f"α with PWM: {alpha}")
print(f"β with PWM: {beta}")
print(f"Estimated U50 : {U50:.2f} m/s")
print("\n")


########### GUMBELL'S FITTING METHOD #############
annual_maxima = sprog.resample('Y', on='timestamp')['wind_speed'].max()
sorted_winds = np.sort(annual_maxima)
N_g = len(sorted_winds)

ranks = np.arange(1, N_g + 1)
Fi = ranks / (N_g + 1)
y = -np.log(-np.log(Fi))


# Linear regression to find alpha and beta 
a, b = np.polyfit(sorted_winds, y, 1)
alpha_g = 1/a
beta_g = -alpha_g*b

U50_g = alpha_g * np.log(T/T_0) + beta_g

print(f"Gumbel Fitting Method:")
print(f"α: {alpha_g}")
print(f"β: {beta_g}")
print(f"Estimated U50 : {U50_g:.2f} m/s")
print("\n")


########### WEIBULL PARAMETER METHOD #############
# I used my old code to re-calculate each weibull parameters 
def weibull_pdf(U, A, k):
    U = np.maximum(U, 1e-6)  # Avoid division by zero
    pdf=(k/U)*(U/A)**k *np.exp(-(U / A)**k)
    return pdf

data_frac = 1 + (var_WS/(mean_WS**2))

def weibull_frac(k):
    gamma_1 = sp.gamma(1 + 1/k)
    gamma_2 = sp.gamma(1 + 2/k)
    return (gamma_2 / gamma_1**2)

def find_best_k(data_frac):
    k_values = np.arange(1, 5, 0.01)  # Extend the search range if needed
    differences = []
    
    for k in k_values:
        theor_frac = weibull_frac(k)
        difference = np.abs(data_frac - theor_frac)
        differences.append(difference)
    
    best_k_index = np.argmin(differences)
    best_k = k_values[best_k_index]
    min_diff = differences[best_k_index]
    
    return best_k, min_diff

k, min_diff = find_best_k(data_frac)
A = mean_WS/sp.gamma(1 + 1/k)

NU = len(windspeed)
cie = 0.438 * NU

beta_w = A * (np.log(cie))**(1/k)
alpha_w = A *(np.log(cie))**(1/k - 1)/k
U50_w = alpha_w * np.log(T/T_0) + beta_w

print(f"Weibull parameter method :")
print(f"α: {alpha_w}")
print(f"β: {beta_w}")
print(f"Estimated U50: {U50_w:.2f} m/s")
print('\n')


########### SIMPLEST METHOD #############
U_mean = windspeed.mean()  
U50_naive = 5 * U_mean

print(f"Simple Method:")
print(f"Mean wind speed (U): {U_mean:.2f} m/s")
print(f"Estimated U50: {U50_naive:.2f} m/s")
print("\n")



################ ESTIMATION FOR NYBORG AND KORSOR ####################
# Take code for gdl and U adjusted from part 1 
kappa = 0.4
z_0_water = 0.02e-2
z_0_land = 3e-2
z = 70
z_hub = 112
A = 1.8
B = 4.5

# Obtaining coriolis parameter from latitude
rot_earth = 7.2921e-5
latitude = 55.3
fc = 2*rot_earth*np.sin(latitude*np.pi/180)

from scipy.optimize import fsolve

def gdl(U, z0, sector):
    u_star = U *kappa/(np.log(z/z_0_water))
    
    epsilon = 1e-10
    U_g = u_star * (np.log(u_star/(fc*z_0_water+epsilon)) - A)/kappa
    
    V_g = - B*u_star/kappa
    G = np.sqrt(U_g ** 2 + V_g ** 2)

    def equation(u_star_guess):
        U_g_guess = u_star_guess * (np.log(u_star_guess / (fc * z0)) - A) / kappa
        V_g_guess = - B * u_star_guess / kappa
        G_guess = np.sqrt(U_g_guess**2 + V_g_guess**2)
        
        return G_guess - G
    
    u_star_guess_initial = 0.01
    u_star_solution = fsolve(equation, u_star_guess_initial)
    
    u_star_final = u_star_solution[0]

    # Compute new wind speed U
    new_U = u_star_final*np.log(z / z0)/kappa
    return new_U


# Split for sectors and get Weibull parameters 
data = {
    "sector": [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330],
    "A": [7.750969, 7.525115, 7.761153, 9.236267, 9.603213, 8.461672, 9.407033, 9.931700, 9.951515, 10.194087, 9.630792, 7.896237],
    "k": [1.900057, 1.894834, 2.046636, 2.311559, 2.316891, 2.064537, 2.157974, 2.468474, 2.553114, 2.563006, 2.200351, 1.895504]}

# Create DataFrame
Weib_parm = pd.DataFrame(data)
Weib_parm_dic = {row['sector']: {'A': row['A'], 'k': row['k']} for _, row in Weib_parm.iterrows()}
bins = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360])
labels = np.arange(0, 360, 30)

sprog['sector'] = pd.cut(sprog['wind_direction'], bins=bins, labels=labels, right=False)
sprog.loc[(sprog['wind_direction'] >= 345) | (sprog['wind_direction'] < 15), 'sector'] = 0

sector_dfs = {sector: sprog[sprog['sector'] == sector] for sector in sprog['sector'].unique()}
sector_dfs.pop(np.nan, None)

# Get the roughness according to sectors 
def get_sector_roughness_dict(site):
    if site == "Nyborg":
        return {sector: z_0_water if sector <= 180 else z_0_land for sector in Weib_parm_dic.keys()}
    elif site == "Korsor":
        return {sector: z_0_land if sector <= 180 else z_0_water for sector in Weib_parm_dic.keys()}
    else:
        raise ValueError("Invalid site name. Choose 'Nyborg' or 'Korsor'.")

sector_roughness_ny = get_sector_roughness_dict("Nyborg")
sector_roughness_ko = get_sector_roughness_dict("Korsor")

# In order to stock the results in dataframe 
data_ny = {'sector': [], 'wind_speed': []}
data_ko = {'sector': [], 'wind_speed': []}

U_ny = []
U_ko = []

# Get U for Nyborg and Korsor 
for sector, df in sector_dfs.items():
    # Get Weibull parameters for the sector
    k = Weib_parm_dic[sector]['k']  
    A = Weib_parm_dic[sector]['A'] 
    z0_ny = sector_roughness_ny[sector] 
    z0_ko = sector_roughness_ko[sector]
    
    # Generate wind speed array 
    wind_speeds = np.sort(df['wind_speed'].unique())
    
    # Should we take only the strongest winds ? 
    U = wind_speeds[-10:] 
    
    # Or otherwise : 
    # U = np.arange(min(wind_speeds), max(wind_speeds), 0.1)
    
    for i in range(len(U) - 1): 
        U_ny.append(gdl(U[i], z0_ny, sector))
        U_ko.append(gdl(U[i], z0_ko, sector))
        
        data_ny['sector'].append(sector)
        data_ny['wind_speed'].append(U_ny)
        
        data_ko['sector'].append(sector)
        data_ko['wind_speed'].append(U_ko)
    
    
############# PWM METHOD ###############
N2 = len(U_ny)
mean_WS_ny = np.mean(U_ny)
mean_WS_ko = np.mean(U_ko)

# Calculate the PWM moments b0 and b1
b0_ny = mean_WS_ny
b1_ny = np.sum((np.arange(1, N2+1) - 1) * U_ny) / (N2 * (N2-1))

b0_ko = mean_WS_ko
b1_ko = np.sum((np.arange(1, N2+1) - 1) * U_ko) / (N2 * (N2-1))


alpha_ny = 2*b1_ny- b0_ny*np.log(2)
beta_ny = b0_ny - gamma * alpha_ny
U50_ny = alpha_ny * np.log(T/T_0) + beta_ny

alpha_ko = 2*b1_ko- b0_ko*np.log(2)
beta_ko = b0_ny - gamma * alpha_ko
U50_ko = alpha_ko * np.log(T/T_0) + beta_ko


print(f"PWM Method for Nyborg:")
print(f"α with PWM: {alpha_ny}")
print(f"β with PWM: {beta_ny}")
print(f"Estimated U50 : {U50_ny:.2f} m/s")
print("\n")


print(f"PWM Method for Korsor:")
print(f"α with PWM: {alpha_ko}")
print(f"β with PWM: {beta_ko}")
print(f"Estimated U50 : {U50_ko:.2f} m/s")
print("\n")


# Visualize strongest winds for the three lcoations 
def ten_extreme_winds(sector_df):
    sector_df_sorted = sector_df.sort_values(by='wind_speed', ascending=False)
    top_10_extreme = sector_df_sorted.head(10)
    return top_10_extreme

df_ny = pd.DataFrame(data_ny)
df_ko = pd.DataFrame(data_ko)

top_10_sprog = {sector: df.nlargest(10, 'wind_speed') for sector, df in sector_dfs.items()}
top_10_ny = ten_extreme_winds(df_ny)
top_10_ko = ten_extreme_winds(df_ko)







import matplotlib.pyplot as plt
from windrose import WindroseAxes

# def assign_directions(df):
#     bins = np.arange(0, 361, 30)
#     central_directions = (bins[:-1] + bins[1:]) / 2 
#     central_directions[-1] = 0  

#     df['wind_direction'] = df['sector'].map(dict(zip(bins[:-1], central_directions)))
#     return df

# top_10_ny = assign_directions(top_10_ny)
# top_10_ko = assign_directions(top_10_ko)

# Function to plot windrose
# def plot_custom_windrose(top_10_sprog, ax, title):
#     # Define the number of bins (sectors)
#     bins = np.linspace(0, 360, 13, endpoint=False)  # 12 bins, each 30 degrees wide
#     wind_hist = np.zeros(len(bins)-1)  # Initialize a zero array to hold the wind speeds per sector
    
#     # Iterate over each sector and accumulate the wind speeds
#     for sector, top_10 in top_10_sprog.items():
#         wind_directions = top_10['wind_direction']
#         wind_speeds = top_10['wind_speed']
        
#         # Compute the histogram for the sector wind speeds
#         hist, _ = np.histogram(wind_directions, bins=bins, weights=wind_speeds)
        
#         # Add this sector's data to the wind_hist array
#         wind_hist += hist

#     # Plot the windrose
#     ax.bar(np.radians(bins[:-1]), wind_hist, width=np.radians(30), bottom=0.0, align='edge', edgecolor='black', linewidth=2)
#     ax.set_title(title)

# fig, axs = plt.subplots(figsize=(10, 10), nrows=1, ncols=1, subplot_kw={'projection': 'polar'})

# plot_custom_windrose(pd.concat(top_10_sprog.values()), axs[0], title="Sprogø - Wind Rose")
# plot_custom_windrose(top_10_ny, axs[1], title="Nyborg - Wind Rose")
# plot_custom_windrose(top_10_ko, axs[2], title="Korsør - Wind Rose")

# # Show the plot
# plt.tight_layout()
# plt.show()








