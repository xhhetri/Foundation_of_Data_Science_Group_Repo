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

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# -------------------------
# Configuration
# -------------------------
DATASET1_PATH = 'dataset1.csv'
DATASET2_PATH = 'dataset2.csv'
FIG_DIR = 'invfind2'
os.makedirs(FIG_DIR, exist_ok=True)

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("colorblind")

# -------------------------
# Load Datasets
# -------------------------
bats = pd.read_csv(DATASET1_PATH)
rats = pd.read_csv(DATASET2_PATH)

# Convert time columns safely
if 'start_time' in bats.columns:
    bats['start_time'] = pd.to_datetime(bats['start_time'], errors='coerce')
if 'time' in rats.columns:
    rats['time'] = pd.to_datetime(rats['time'], errors='coerce')

# Binary column for rat presence
if 'seconds_after_rat_arrival' in bats.columns:
    bats['rat_present'] = bats['seconds_after_rat_arrival'].notnull().astype(int)
else:
    bats['rat_present'] = 0

# -------------------------
# Map months to seasons
# -------------------------
def month_to_season(month):
    if month in ['December','January','February']:
        return 'Winter'
    elif month in ['March','April','May']:
        return 'Spring'
    elif month in ['June','July','August']:
        return 'Summer'
    elif month in ['September','October','November']:
        return 'Autumn'
    else:
        return 'Unknown'

if 'month' in rats.columns:
    rats['season'] = rats['month'].apply(month_to_season)
else:
    rats['season'] = 'Unknown'

# -------------------------
# Step 1: Seasonal Analysis - Bat Vigilance
# -------------------------
plt.figure(figsize=(8,6))
sns.boxplot(x='season', y='bat_landing_to_food', hue='rat_present', data=bats)
plt.ylabel('Time to Approach Food (s)')
plt.title('Seasonal Variation in Bat Vigilance')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR,'seasonal_vigilance.png'), dpi=300)
plt.close()

# -------------------------
# Step 2: Seasonal Analysis - Bat Risk Behavior
# -------------------------
plt.figure(figsize=(8,6))
sns.countplot(x='season', hue='risk', data=bats)
plt.title('Seasonal Variation in Bat Risk Behavior')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR,'seasonal_risk_behavior.png'), dpi=300)
plt.close()

# -------------------------
# Step 3: Seasonal Analysis - Rat Activity
# -------------------------
plt.figure(figsize=(8,6))
sns.boxplot(x='season', y='rat_arrival_number', data=rats)
plt.ylabel('Number of Rat Arrivals per 30 min')
plt.title('Seasonal Variation in Rat Arrivals')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR,'seasonal_rat_arrivals.png'), dpi=300)
plt.close()

# Bat landings vs rat activity by season
plt.figure(figsize=(8,6))
sns.scatterplot(x='rat_arrival_number', y='bat_landing_number', hue='season', data=rats)
plt.xlabel('Number of Rat Arrivals per 30 min')
plt.ylabel('Number of Bat Landings per 30 min')
plt.title('Bat Landings vs Rat Activity by Season')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR,'bat_vs_rat_activity.png'), dpi=300)
plt.close()

# -------------------------
# Step 4: Summary Tables
# -------------------------
summary_vigilance = bats.groupby(['season','rat_present'])['bat_landing_to_food'].mean().reset_index()
print("\nMean Bat Vigilance by Season and Rat Presence:\n", summary_vigilance)

risk_summary = bats.groupby(['season','risk']).size().unstack(fill_value=0)
risk_summary_percent = risk_summary.div(risk_summary.sum(axis=1), axis=0) * 100
print("\nRisk Behavior (%) by Season:\n", risk_summary_percent)

# -------------------------
# Step 5: Existing Analyses (optional ANOVA / interaction)
# -------------------------
if 'bat_landing_to_food' in bats.columns:
    # ANOVA for foraging delay across seasons
    seasons = ['Winter','Spring','Summer','Autumn']
    groups = [bats[bats['season']==s]['bat_landing_to_food'].dropna() for s in seasons]
    if all([len(g)>0 for g in groups]):
        from scipy.stats import f_oneway
        f_stat, p_val = f_oneway(*groups)
        print(f"\nANOVA for foraging delay across seasons: F={f_stat:.3f}, p={p_val:.4f}")

print("\nAll figures saved in folder:", FIG_DIR)
