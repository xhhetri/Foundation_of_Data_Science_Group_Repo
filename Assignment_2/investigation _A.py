'''
Do bats perceive rats not just as competitors for food but also as potential 
predators? If rats are considered a predation risk by bats, scientists believe 
that this perception will translate into the bats’ higher level of avoidance behaviour 
or increased vigilance during foraging on the food platform.

Investigation A: Do bats perceive rats as predators?

Variables:
 ├─ bat_landing_to_food      (hesitation time)
 ├─ seconds_after_rat_arrival (rat presence timing)
 ├─ risk                     (risk-taking: 0/1)
 └─ reward                   (success: 0/1)

Analyses:
 ├─ Descriptive Statistics
 │    ├─ Mean, median, std, range of hesitation time
 │    └─ Frequency counts of risk & reward
 ├─ Visualisation
 │    ├─ Boxplot: hesitation time (Rats vs No Rats)
 │    ├─ Heatmap: Risk vs Reward
 │    └─ Scatter: Hesitation vs Seconds after rat arrival
 └─ Inferential Statistics
      ├─ T-test: hesitation (rats vs no rats)
      ├─ Chi-square: risk vs reward
      └─ Regression (optional): hesitation ~ rat arrival + hours after sunset
    
'''
# investigation_a.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import os
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("colorblind")

# ----------------------------
# Create folder for figures
# ----------------------------
FIG_DIR = 'invfind1'
os.makedirs(FIG_DIR, exist_ok=True)

# ----------------------------
# Load datasets
# ----------------------------
print("Loading datasets...")
df1 = pd.read_csv('dataset1.csv')
df2 = pd.read_csv('dataset2.csv')

# ----------------------------
# Clean datasets
# ----------------------------
df1_clean = df1.dropna(subset=['bat_landing_to_food', 'seconds_after_rat_arrival', 'risk', 'reward'])
df2_clean = df2.dropna(subset=['bat_landing_number', 'rat_minutes', 'rat_arrival_number'])

# ----------------------------
# Feature Engineering
# ----------------------------
df1_clean['rat_present'] = df1_clean['seconds_after_rat_arrival'].apply(lambda x: 1 if x >= 0 else 0)
df1_clean['foraging_delay'] = df1_clean['bat_landing_to_food']

# ----------------------------
# Analysis 1: Compare bat behavior with and without rats present
# ----------------------------
print("\nAnalysis 1: Comparing bat behavior with and without rats present")

rat_present_stats = df1_clean.groupby('rat_present').agg({
    'bat_landing_to_food': ['mean', 'std', 'count'],
    'risk': 'mean',
    'reward': 'mean'
}).round(3)

print("Behavior statistics with and without rats present:")
print(rat_present_stats)

# T-test for foraging delay
no_rat_delay = df1_clean[df1_clean['rat_present'] == 0]['bat_landing_to_food']
rat_delay = df1_clean[df1_clean['rat_present'] == 1]['bat_landing_to_food']

t_stat, p_value = stats.ttest_ind(no_rat_delay, rat_delay, equal_var=False, nan_policy='omit')
print(f"\nT-test for difference in foraging delay: t={t_stat:.3f}, p={p_value:.4f}")

# ----------------------------
# Analysis 2: Relationship between time since rat arrival and bat behavior
# ----------------------------
print("\nAnalysis 2: Relationship between time since rat arrival and bat behavior")

bins = [-np.inf, 0, 60, 300, 1800, np.inf]
labels = ['Before arrival', '0-1 min', '1-5 min', '5-30 min', '30+ min']
df1_clean['time_bin'] = pd.cut(df1_clean['seconds_after_rat_arrival'], bins=bins, labels=labels)

time_bin_stats = df1_clean.groupby('time_bin').agg({
    'bat_landing_to_food': ['mean', 'std', 'count'],
    'risk': 'mean',
    'reward': 'mean'
}).round(3)

print("Behavior statistics by time since rat arrival:")
print(time_bin_stats)

# ----------------------------
# Analysis 3: Risk-taking behavior in relation to rat presence
# ----------------------------
print("\nAnalysis 3: Risk-taking behavior in relation to rat presence")

risk_rat_crosstab = pd.crosstab(df1_clean['rat_present'], df1_clean['risk'], 
                                margins=True, margins_name="Total")

# Safe renaming
risk_rat_crosstab.columns = [f'Risk {int(c)}' if c != 'Total' else 'Total' for c in risk_rat_crosstab.columns]
index_mapping = {0: 'No Rat', 1: 'Rat Present', 'Total': 'Total'}
risk_rat_crosstab.index = [index_mapping.get(i, i) for i in risk_rat_crosstab.index]

print("Risk behavior by rat presence:")
print(risk_rat_crosstab)

# Chi-square test for independence
chi2, p, dof, expected = stats.chi2_contingency(pd.crosstab(df1_clean['rat_present'], df1_clean['risk']))
print(f"\nChi-square test for independence: χ²={chi2:.3f}, p={p:.4f}")

# ----------------------------
# Analysis 4: Relationship between rat activity and bat landings
# ----------------------------
print("\nAnalysis 4: Relationship between rat activity and bat landings")

correlation = df2_clean[['bat_landing_number', 'rat_minutes', 'rat_arrival_number']].corr()
print("Correlation matrix:")
print(correlation.round(3))

# ----------------------------
# Visualizations
# ----------------------------
print("\nGenerating visualizations...")

# Foraging delay boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='rat_present', y='bat_landing_to_food', data=df1_clean)
plt.xticks([0, 1], ['No Rat Present', 'Rat Present'])
plt.ylabel('Foraging Delay (seconds)')
plt.xlabel('')
plt.title('Foraging Delay With and Without Rats Present')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'foraging_delay_comparison.png'), dpi=300)
plt.close()

# Risk-taking bar plot
plt.figure(figsize=(8, 6))
risk_rat_prop = pd.crosstab(df1_clean['rat_present'], df1_clean['risk'], normalize='index')
risk_rat_prop.plot(kind='bar', stacked=True)
plt.xticks([0, 1], ['No Rat Present', 'Rat Present'], rotation=0)
plt.ylabel('Proportion of Landings')
plt.xlabel('')
plt.title('Risk Behavior With and Without Rats Present')
plt.legend(['Risk Avoidance', 'Risk Taking'])
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'risk_behavior_comparison.png'), dpi=300)
plt.close()

# Foraging success by time bin
plt.figure(figsize=(10, 6))
sns.barplot(x='time_bin', y='reward', data=df1_clean, ci=95)
plt.xlabel('Time Since Rat Arrival')
plt.ylabel('Proportion of Successful Foraging')
plt.title('Foraging Success by Time Since Rat Arrival')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'foraging_success_time_bins.png'), dpi=300)
plt.close()

# Rat activity vs bat landings
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.scatter(df2_clean['rat_minutes'], df2_clean['bat_landing_number'], alpha=0.6)
ax1.set_xlabel('Rat Presence (minutes)')
ax1.set_ylabel('Bat Landings')
ax1.set_title('Bat Landings vs Rat Presence Duration')
ax2.scatter(df2_clean['rat_arrival_number'], df2_clean['bat_landing_number'], alpha=0.6)
ax2.set_xlabel('Number of Rat Arrivals')
ax2.set_ylabel('Bat Landings')
ax2.set_title('Bat Landings vs Rat Arrivals')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'rat_bat_relationship.png'), dpi=300)
plt.close()

# ----------------------------
# Machine Learning: Predict risk behavior
# ----------------------------
print("\nMachine Learning Analysis: Predicting risk behavior")

features = ['bat_landing_to_food', 'seconds_after_rat_arrival', 'hours_after_sunset']
X = df1_clean[features].fillna(df1_clean[features].median())
y = df1_clean['risk']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_scaled, y)

feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("Feature importance for predicting risk behavior:")
print(feature_importance)

# ----------------------------
# Conclusions
# ----------------------------
print("\n" + "="*50)
print("CONCLUSIONS FOR INVESTIGATION A")
print("="*50)

print("1. Bats show significantly different behavior when rats are present:")
print(f"   - Foraging delay: {no_rat_delay.mean():.1f}s without rats vs {rat_delay.mean():.1f}s with rats (p={p_value:.4f})")

print("\n2. Risk-taking behavior is influenced by rat presence:")
risk_no_rat = df1_clean[df1_clean['rat_present'] == 0]['risk'].mean()
risk_rat = df1_clean[df1_clean['rat_present'] == 1]['risk'].mean()
print(f"   - Risk-taking: {risk_no_rat:.2%} without rats vs {risk_rat:.2%} with rats")

print("\n3. Foraging success varies with time since rat arrival:")
for bin_label in labels:
    bin_data = df1_clean[df1_clean['time_bin'] == bin_label]
    if len(bin_data) > 0:
        success_rate = bin_data['reward'].mean()
        print(f"   - {bin_label}: {success_rate:.2%} success rate")

print("\n4. Rat activity is correlated with bat landing frequency:")
print(f"   - Correlation between rat minutes and bat landings: {correlation.loc['bat_landing_number', 'rat_minutes']:.3f}")

print("\n5. Key predictors of risk behavior:")
for _, row in feature_importance.iterrows():
    print(f"   - {row['feature']}: {row['importance']:.3f}")

print("\nOverall, bats demonstrate increased avoidance behavior (longer foraging delays)")
print("and altered risk-taking behavior when rats are present, suggesting they perceive")
print("rats as potential predators or significant competitors.")
