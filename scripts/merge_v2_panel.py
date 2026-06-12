"""Merge v2 policy scores into master panel."""
import pandas as pd
import numpy as np

ps2 = pd.read_csv("output/policy_scores_panel_v2.csv")
mp = pd.read_csv("output/master_panel_data_v2.csv")

# Keep only Gaoyang (130628) from master panel (other counties don't have enterprise data)
gaoyang = mp[mp['County_Code'].astype(str) == '130628'].copy()

# Select key v2 columns
v2_cols = ['County_Code', 'Year', 'policy_intensity_total', 'policy_intensity_concentration',
           'equipment_sum', 'environment_sum', 'ecommerce_sum', 'brandquality_sum',
           'cluster_sum', 'finance_sum', 'education_sum',
           'equipment_index', 'environment_index', 'ecommerce_index',
           'brandquality_index', 'cluster_index', 'finance_index', 'education_index',
           'total_chunks', 'scored', 'keyword_passed', 'sample_reliable']

gy_ps2 = ps2[ps2['County_Code'].astype(str) == '130628'][v2_cols].copy()

# Merge
merged = gaoyang.merge(gy_ps2, on=['County_Code', 'Year'], how='left', suffixes=('', '_v2'))

# Generate derived variables
merged['time'] = merged['Year'] - 2000
merged['post2017'] = (merged['Year'] >= 2017).astype(int)
merged['ln_textile_firms'] = np.log(merged['textile_firms'])

# Diff variables
merged['d_textile'] = merged['textile_firms'].diff()
merged['d_policy_total'] = merged['policy_intensity_total'].diff()
merged['d_policy_conc'] = merged['policy_intensity_concentration'].diff()

merged.to_csv("output/master_panel_v2.csv", index=False, encoding='utf-8-sig')
print(f"Merged master_panel_v2.csv: {len(merged)} rows, {len(merged.columns)} columns")
print(f"Year range: {merged['Year'].min()}-{merged['Year'].max()}")
print(f"Policy total non-zero: {(merged['policy_intensity_total']>0).sum()}/{len(merged)}")
