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
# Analysis of bat behavior in presence vs absence of rats (Investigation A)
# Using dataset1.csv only

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, chi2_contingency
import warnings
warnings.filterwarnings('ignore')


# Step 1: Load Dataset1

bats = pd.read_csv('dataset1.csv')

# Convert time columns
bats['start_time'] = pd.to_datetime(bats['start_time'], errors='coerce')
bats['rat_period_start'] = pd.to_datetime(bats['rat_period_start'], errors='coerce')
bats['rat_period_end'] = pd.to_datetime(bats['rat_period_end'], errors='coerce')


# Step 2: Feature Engineering

# Binary indicator for rat presence during bat landing
bats['rat_present'] = bats['seconds_after_rat_arrival'].notnull().astype(int)


# Step 3: Descriptive Analysis

# Vigilance (time to approach food)
plt.figure(figsize=(8,6))
sns.boxplot(x='rat_present', y='bat_landing_to_food', data=bats)
plt.xticks([0,1], ['No Rats', 'Rats Present'])
plt.ylabel('Time to Approach Food (s)')
plt.title('Bat Vigilance vs Rat Presence')
plt.savefig('vigilance_rat_presence.png')
plt.show()

# Risk-taking behavior
plt.figure(figsize=(6,5))
sns.countplot(x='risk', hue='rat_present', data=bats)
plt.xlabel('Risk Behavior (0=Avoid, 1=Take)')
plt.ylabel('Count')
plt.title('Bat Risk Behavior vs Rat Presence')
plt.legend(['No Rats', 'Rats Present'])
plt.savefig('risk_behavior_rat_presence.png')
plt.show()


# Step 4: Inferential Analysis

# T-test for vigilance
vigilance_rats = bats[bats['rat_present']==1]['bat_landing_to_food'].dropna()
vigilance_no_rats = bats[bats['rat_present']==0]['bat_landing_to_food'].dropna()
t_stat, p_val = ttest_ind(vigilance_rats, vigilance_no_rats)
print(f"T-test for vigilance: t={t_stat:.2f}, p={p_val:.4f}")

# Chi-square test for risk behavior
risk_table = pd.crosstab(bats['risk'], bats['rat_present'])
chi2, p_risk, dof, expected = chi2_contingency(risk_table)
print(f"Chi-square test for risk behavior: chi2={chi2:.2f}, p={p_risk:.4f}")


# Step 5: Summary Tables

summary_vigilance = bats.groupby('rat_present')['bat_landing_to_food'].mean().reset_index()
print("\nMean Bat Vigilance by Rat Presence:\n", summary_vigilance)

risk_summary = bats.groupby('rat_present')['risk'].value_counts().unstack(fill_value=0)
risk_summary_percent = risk_summary.div(risk_summary.sum(axis=1), axis=0) * 100
print("\nRisk Behavior (%) by Rat Presence:\n", risk_summary_percent)
