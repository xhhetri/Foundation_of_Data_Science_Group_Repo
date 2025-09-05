'''
Do the behaviours described above change following seasonal changes? 
It is known that in winter, alternative food sources are scarce and 
rat encounters are less frequent. In spring, food is more abundant and 
rat encounters are more common.

Investigation B: Do behaviours change with season?

Variables:
 ├─ season                   (Winter/Spring)
 ├─ bat_landing_to_food      (hesitation)
 ├─ risk                     (risk-taking: 0/1)
 ├─ reward                   (success: 0/1)
 ├─ rat_arrival_number       (Dataset2)
 └─ bat_landing_number       (Dataset2)

Analyses:
 ├─ Descriptive Statistics
 │    ├─ Hesitation stats per season (mean, median, std)
 │    ├─ Risk & reward counts per season
 │    └─ Rat & bat activity averages per season
 ├─ Visualisation
 │    ├─ Boxplot: hesitation by season
 │    ├─ Bar chart: Rat & Bat activity by season
 │    └─ Heatmap: Risk vs Reward by season
 └─ Inferential Statistics
      ├─ ANOVA/Kruskal-Wallis: hesitation across seasons
      ├─ Chi-square: risk vs reward by season
      └─ Optional regression: rat arrivals → bat landings/food availability
      
'''

# investigation_b.py
# Seasonal variation in bat and rat behavior (Investigation B)
# Using dataset1.csv and dataset2.csv

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# Step 1: Load Datasets

bats = pd.read_csv('dataset1.csv')
rats = pd.read_csv('dataset2.csv')

# Convert time columns
bats['start_time'] = pd.to_datetime(bats['start_time'], errors='coerce')
rats['time'] = pd.to_datetime(rats['time'], errors='coerce')

# Binary column for rat presence
bats['rat_present'] = bats['seconds_after_rat_arrival'].notnull().astype(int)


# Step 2: Map months to seasons for rats dataset

def month_to_season(month):
    if month in ['December', 'January', 'February']:
        return 'Winter'
    elif month in ['March', 'April', 'May']:
        return 'Spring'
    elif month in ['June', 'July', 'August']:
        return 'Summer'
    elif month in ['September', 'October', 'November']:
        return 'Autumn'
    else:
        return 'Unknown'

rats['season'] = rats['month'].apply(month_to_season)


# Step 3: Seasonal Analysis - Bat Vigilance

plt.figure(figsize=(8,6))
sns.boxplot(x='season', y='bat_landing_to_food', hue='rat_present', data=bats)
plt.ylabel('Time to Approach Food (s)')
plt.title('Seasonal Variation in Bat Vigilance')
plt.savefig('seasonal_vigilance.png')
plt.show()


# Step 4: Seasonal Analysis - Bat Risk Behavior

plt.figure(figsize=(8,6))
sns.countplot(x='season', hue='risk', data=bats)
plt.title('Seasonal Variation in Bat Risk Behavior')
plt.savefig('seasonal_risk_behavior.png')
plt.show()


# Step 5: Seasonal Analysis - Rat Activity

plt.figure(figsize=(8,6))
sns.boxplot(x='season', y='rat_arrival_number', data=rats)
plt.ylabel('Number of Rat Arrivals per 30 min')
plt.title('Seasonal Variation in Rat Arrivals')
plt.savefig('seasonal_rat_arrivals.png')
plt.show()

# Bat landings vs rat activity by season
plt.figure(figsize=(8,6))
sns.scatterplot(x='rat_arrival_number', y='bat_landing_number', hue='season', data=rats)
plt.xlabel('Number of Rat Arrivals per 30 min')
plt.ylabel('Number of Bat Landings per 30 min')
plt.title('Bat Landings vs Rat Activity by Season')
plt.savefig('bat_vs_rat_activity.png')
plt.show()


# Step 6: Summary Tables

# Mean vigilance by season and rat presence
summary_vigilance = bats.groupby(['season','rat_present'])['bat_landing_to_food'].mean().reset_index()
print("\nMean Bat Vigilance by Season and Rat Presence:\n", summary_vigilance)

# Risk behavior percentages by season
risk_summary = bats.groupby(['season','risk']).size().unstack(fill_value=0)
risk_summary_percent = risk_summary.div(risk_summary.sum(axis=1), axis=0) * 100
print("\nRisk Behavior (%) by Season:\n", risk_summary_percent)
