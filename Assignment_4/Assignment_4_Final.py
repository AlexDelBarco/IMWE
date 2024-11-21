### IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

def power_curve(U):
    if U<=Urp:
        return Prated*(U/Urp)**3
    elif Urp<U<25:
        return Prated
    else :
        return 0
    return False

def weibull_pdf(U,k,A):
    U = np.maximum(U, 1e-6)
    pdf = (k/U)*((U/A)**k) * np.exp(-(U/A)**k)
    return pdf


def ff_i(sector):
    fi = len(sector_dfs[sector])/len(sprog)
    return fi


def calculate_AEP_sprogo(sector_dfs, Weib_parm_dic):
    E_sector = {}
    AEP = 0 

    # Wind speed range (from 0 to 40 m/s)
    #U = np.arange(0, 40.1, 0.1)

    for key, df in sector_dfs.items():
        k = Weib_parm_dic[key]['k']  
        A = Weib_parm_dic[key]['A']  
        
        # Extract unique wind speeds from the sector data
        wind_speeds = np.sort(df['wind_speed'].unique())
        U = np.arange(min(wind_speeds), max(wind_speeds), 0.1)
    
        integral_f = []  
        delta_Us = []    
        pdfs = []        
        powers = []      
        
        for i in range(len(U) - 1): 
            delta_U = abs(U[i + 1] - U[i])  # Difference in wind speed has to be positive
            delta_Us.append(delta_U)
            midpoint_U = (U[i] + U[i + 1]) / 2

            pdf_value = weibull_pdf(midpoint_U, k, A)
            pdfs.append(pdf_value)
            power_value = power_curve(midpoint_U)
            powers.append(power_value)
            integral = pdf_value * power_value * delta_U
            integral_f.append(integral)

        integral_T = sum(integral_f)
        fi = ff_i(key)  
        E_sector[key] = T * fi * integral_T 
        AEP += E_sector[key]

    return AEP, E_sector

AEP_sprogo, E_sector_sprogo = calculate_AEP_sprogo(sector_dfs, Weib_parm_dic)
print(f"Total AEP at Sprogo in MWh: {AEP_sprogo:.3f}")


####################### AEP at the two other sites ###################################
# Calculate u_star, then geostrophic wind, then new u_star and finally new wind speed for each previous data
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
    
    U_g = u_star * (np.log(u_star/(fc*z_0_water)) - A)/kappa
    V_g = - B*u_star/kappa
    G = np.sqrt(U_g ** 2 + V_g ** 2)

    def equation(u_star_guess):
        U_g_guess = u_star_guess * (np.log(u_star_guess / (fc * z0)) - A) / kappa
        V_g_guess = - B * u_star_guess / kappa
        G_guess = np.sqrt(U_g_guess**2 + V_g_guess**2)
        
        return G_guess - G
    
    
    u_star_guess_initial = 0.001
    u_star_solution = fsolve(equation, u_star_guess_initial)
    
    u_star_final = u_star_solution[0]
    
    # # Ensure that u_star doesn't exceed reasonable limits
    # max_u_star = 1
    # if u_star_final > max_u_star:
    #     u_star_final = max_u_star
    
    # Compute new wind speed U
    new_U = u_star_final*np.log(z / z0)/kappa   # change here, extrapolete Weibull but not U
    return new_U


# Now we have to calculate AEP with new wind speeds 
# Get the roughness according to sectors 
def get_sector_roughness_dict(site):
    if site == "Nyborg":
        return {sector: z_0_water if sector <= 180 else z_0_land for sector in Weib_parm_dic.keys()}
    elif site == "Korsor":
        return {sector: z_0_land if sector <= 180 else z_0_water for sector in Weib_parm_dic.keys()}
    else:
        raise ValueError("Invalid site name. Choose 'Nyborg' or 'Korsor'.")

# Generate the dictionaries
sector_roughness_ny = get_sector_roughness_dict("Nyborg")
sector_roughness_ko = get_sector_roughness_dict("Korsor")


def calculate_site_AEP(sector_dfs, sector_roughness, Weib_dic):   
    AEP = 0
    E_sector = {}
    
    for sector, df in sector_dfs.items():
        k = Weib_dic[sector]['k']   # Take the new ones
        A = Weib_dic[sector]['A'] # Take the new ones
        z0 = sector_roughness[sector]

        wind_speeds = np.sort(df['wind_speed'].unique())
        U = np.arange(min(wind_speeds), max(wind_speeds), 0.1)
        
        # Old U distribution ; 
        #U = np.arange(0, 41, 0.5)
        integral_f = []
        
        
        for i in range(len(U) - 1): 
            delta_U = abs(U[i + 1] - U[i])  
            midpoint_U = (U[i] + U[i + 1]) / 2
            
            U_new = gdl(midpoint_U, z0, sector) 
            #print(f"Midpoint wind speed: {midpoint_U}, Adjusted wind speed (U_new): {U_new}")
            
            pdf_value_new = weibull_pdf(U_new, k, A)
            power_value_new = power_curve(U_new)
            
            #print(f"PDF value: {pdf_value_new}, Power value: {power_value_new}")
            
            integral_new = pdf_value_new * power_value_new * delta_U
            integral_f.append(integral_new)

        integral_T = sum(integral_f)
        fi = ff_i(sector)
        E_sector[sector] = T * fi * integral_T  
        AEP += E_sector[sector]  

    return AEP, E_sector

# Calculate AEP for Nyborg and Korsor
AEP_Nyborg, E_sector_Nyborg = calculate_site_AEP(sector_dfs, sector_roughness_ny, Weib_parm_dic)
AEP_Korsor, E_sector_Korsor = calculate_site_AEP(sector_dfs, sector_roughness_ko, Weib_parm_dic)

print(f"Total AEP at Nyborg in MWh: {AEP_Nyborg:.3f}")
print(f"Total AEP at Korsor in MWh: {AEP_Korsor:.3f}")


# "Q1a"
# # Define the reduced k (120m)
# Weib_parm_dic_reduced = {sector: {'A': values['A'], 'k': values['k'] * 0.85} for sector, values in Weib_parm_dic.items()}

# AEP_ny_120m = calculate_site_AEP(sector_dfs, sector_roughness_ny, Weib_parm_dic_reduced)
# AEP_ko_120m = calculate_site_AEP(sector_dfs, sector_roughness_ko, Weib_parm_dic_reduced)
# AEP_sprogo_120m, E_sector_sprogo_120m = calculate_AEP_sprogo(sector_dfs, Weib_parm_dic_reduced)

# print(f"Total AEP at Sprogo for 120m (in MWh): {AEP_sprogo_120m:.3f}")
# print(f"Total AEP at Nyborg for 120m (in MWh): {AEP_ny_120m[0]:.2f}")
# print(f"Total AEP at Korsor for 120m (in MWh): {AEP_ko_120m[0]:.2f}")

# # Calculate the percentage difference
# percent_difference_sprog = 100*(AEP_sprogo_120m - AEP_sprogo) /AEP_sprogo 
# percent_difference_ny = 100*(AEP_ny_120m[0] - AEP_Nyborg) /AEP_Nyborg 
# percent_difference_ko = 100*(AEP_ko_120m[0] - AEP_Korsor) /AEP_Korsor 

# print(f"Percentage difference in AEP due to 15% reduction in Sprogo: {abs(percent_difference_sprog):.2f}%")
# print(f"Percentage difference in AEP due to 15% reduction in Nyborg: {abs(percent_difference_ny):.2f}%")
# print(f"Percentage difference in AEP due to 15% reduction in Korsor: {abs(percent_difference_ko):.2f}%")
# print("\n")


"Q1b"
# Group timestamp by year
year_counts = sprog['year'].value_counts().sort_index()
print("Number of Records per Year:")
print(year_counts)


# The data are really different, so have to select what we find a "complete" year
threshold = year_counts.max() * 0.65
sprog_filtered = sprog[sprog['year'].isin(year_counts[year_counts >= threshold].index)]
avg_wind_speed_per_year = sprog_filtered.groupby('year')['wind_speed'].mean()

# Now to find most windy and weakest, we have to calculate mean wind speed
most_windy_year = avg_wind_speed_per_year.idxmax() 
least_windy_year = avg_wind_speed_per_year.idxmin()

print(f"Most windy year: {most_windy_year}")
print(f"Least windy year: {least_windy_year}")


# Calculate AEP for Sprogo, Nyborg and then Korsor 
sprog_most_windy = sprog[sprog['year'] == most_windy_year]
sprog_least_windy = sprog[sprog['year'] == least_windy_year]

# Have to recreate the sectors of question 1 
sprog_most_windy.loc[:, 'sector'] = pd.cut(sprog_most_windy['wind_direction'], bins=bins, labels=labels, right=False)
sprog_most_windy.loc[(sprog_most_windy['wind_direction'] >= 345) | (sprog_most_windy['wind_direction'] < 15), 'sector'] = 0
sprog_least_windy.loc[:, 'sector'] = pd.cut(sprog_least_windy['wind_direction'], bins=bins, labels=labels, right=False)
sprog_least_windy.loc[(sprog_least_windy['wind_direction'] >= 345) | (sprog_least_windy['wind_direction'] < 15), 'sector'] = 0

sector_dfs_most = {sector: sprog_most_windy[sprog_most_windy['sector'] == sector] for sector in sprog_most_windy['sector'].unique()}
sector_dfs_most.pop(np.nan, None)
sector_dfs_least = {sector: sprog_least_windy[sprog_least_windy['sector'] == sector] for sector in sprog_least_windy['sector'].unique()}
sector_dfs_least.pop(np.nan, None)


# Question : do we need to adjust the T in E_i product ? Or keep 1 year ? 
AEP_most_sprogo, _ = calculate_AEP_sprogo(sector_dfs_most, Weib_parm_dic)
AEP_least_sprogo, _ = calculate_AEP_sprogo(sector_dfs_least, Weib_parm_dic)
AEP_most_Nyborg, E_sector_Nyborg = calculate_site_AEP(sector_dfs_most, sector_roughness_ny, Weib_parm_dic)
AEP_most_Korsor, E_sector_Korsor = calculate_site_AEP(sector_dfs_most, sector_roughness_ko, Weib_parm_dic)
AEP_least_Nyborg, E_sector_Nyborg = calculate_site_AEP(sector_dfs_least, sector_roughness_ny, Weib_parm_dic)
AEP_least_Korsor, E_sector_Korsor = calculate_site_AEP(sector_dfs_least, sector_roughness_ko, Weib_parm_dic)

print(f"Total AEP in the most windy year at Sprogo in MWh: {AEP_most_sprogo:.3f}")
print(f"Total AEP in the least windy year at Sprogo in MWh: {AEP_least_sprogo:.3f}")
print(f"Total AEP in most windy year at Nyborg in MWh: {AEP_most_Nyborg:.3f}")
print(f"Total AEP in most windy year at Korsor in MWh: {AEP_most_Korsor:.3f}")
print(f"Total AEP in least windy year at Nyborg in MWh: {AEP_least_Nyborg:.3f}")
print(f"Total AEP in least windy year at Korsor in MWh: {AEP_least_Korsor:.3f}")



"Q1b i)"
# "Mean AEP" are above
diff_most_sprogo = 100*(AEP_most_sprogo- AEP_sprogo)/AEP_sprogo
diff_least_sprogo = 100*(AEP_least_sprogo - AEP_sprogo)/AEP_sprogo
diff_most_ny = 100*(AEP_most_Nyborg - AEP_Nyborg)/AEP_Nyborg
diff_least_ny = 100*(AEP_least_Nyborg - AEP_Nyborg)/AEP_Nyborg
diff_most_ko = 100*(AEP_most_Korsor - AEP_Korsor)/AEP_Korsor
diff_least_ko = 100*(AEP_least_Korsor - AEP_Korsor)/AEP_Korsor

print(f"Percentage diff most Sprogo: {diff_most_sprogo:.3f}")
print(f"Percentage diff least Sprogo: {diff_least_sprogo:.3f}")
print(f"Percentage diff most Nyborg: {diff_most_ny:.3f}")
print(f"Percentage diff least Nyborg: {diff_least_ny:.3f}")
print(f"Percentage diff most Korsor: {diff_most_ko:.3f}")
print(f"Percentage diff least Korsor: {diff_least_ko:.3f}")



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

print(('End'))



# print('The End')



