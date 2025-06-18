# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 11:44:37 2025

@author: pnzuza
"""


import pandas as pd
import numpy as np
from scipy import stats
import scikit_posthocs as sp

# Read data from CSV
original_df = pd.read_csv('C:/Users/pnzuza/Downloads/revison.csv')

# Define response variables (all columns except 'Damage', 'Site', 'Month')
response_vars = [col for col in original_df.columns if col not in ['Damage', 'Site', 'Month']]

# Formatting function for cleaner p-value display
def format_p_value(p_value):
    """Format p-value based on significance criteria"""
    if isinstance(p_value, str) and p_value == "<0.001":
        return "<0.001"
    try:
        p = float(p_value)
        if p > 0.05:
            return f"{p:.2f}"
        elif 0.001 <= p <= 0.05:
            return f"{p:.3f}"
        elif p < 0.001:
            return "<0.001"
    except:
        return str(p_value)

# Function to generate compact letter display (simplified version)
def generate_cld(tukey):
    groups = tukey.groupsunique
    n = len(groups)
    reject = tukey.reject
    nonsig = np.zeros((n, n), dtype=bool)
    k = 0
    for i in range(n):
        for j in range(i+1, n):
            nonsig[i, j] = not reject[k]
            nonsig[j, i] = nonsig[i, j]
            k += 1
    visited = [False] * n
    letters = []
    current_letter = 97  # ASCII code for 'a'
    for i in range(n):
        if not visited[i]:
            component = []
            stack = [i]
            while stack:
                node = stack.pop()
                if not visited[node]:
                    visited[node] = True
                    component.append(node)
                    for j in range(n):
                        if nonsig[node, j] and not visited[j]:
                            stack.append(j)
            for idx in component:
                letters.append((groups[idx], chr(current_letter)))
            current_letter += 1
    cld_dict = {group: letter for group, letter in letters}
    return cld_dict

# Loop through each response variable
for var in response_vars:
    print(f"\n\n{'='*50}\nProcessing response variable: {var}\n{'='*50}")
    df = original_df.copy()
    
    # Perform Levene’s test for homogeneity of variances
    groups = [df[df['Damage'] == d][var] for d in df['Damage'].unique() if len(df[df['Damage'] == d][var]) > 1]

    if len(groups) < 2:
        print(f"Skipping {var}: Not enough groups for Levene's test.")
        continue
    try:
        levene_stat, levene_p = stats.levene(*groups)
    except Exception as e:
        print(f"Skipping Levene's test for {var}: {e}")
        continue
    
    print(f"Levene’s test p-value for {var}: {format_p_value(levene_p)}")
    
    best_transformation = None
    best_p = 1
    best_transformed_data = None
    
    if levene_p < 0.05:
        print("Variance is not homogeneous. Trying transformations...")
        transformations = {
            'log': np.log,
            'sqrt': np.sqrt,
            'reciprocal': lambda x: 1 / x,
            'boxcox': lambda x: stats.boxcox(x + 1 - np.min(x))[0],
            'power2': lambda x: np.power(x, 2)
        }
        
        for name, func in transformations.items():
            try:
                transformed = func(df[var])
                if np.isnan(transformed).any() or np.isinf(transformed).any():
                    raise ValueError(f"{name} transformation resulted in invalid values")
                transformed_groups = [transformed[df['Damage'] == d] for d in df['Damage'].unique()]
                t_stat, t_p = stats.levene(*transformed_groups)
                print(f"{name} transformation Levene p-value: {format_p_value(t_p)}")
                if t_p > 0.05 and t_p < best_p:
                    best_p = t_p
                    best_transformation = name
                    best_transformed_data = transformed
            except Exception as e:
                print(f"{name} transformation failed for {var}: {e}")
        
        if best_transformation:
            print(f"Using {best_transformation} transformation for {var}")
            df[var] = best_transformed_data
            transformed_groups = [df[var][df['Damage'] == d] for d in df['Damage'].unique()]
            levene_stat, levene_p = stats.levene(*transformed_groups)
            print(f"Levene’s test p-value after transformation: {format_p_value(levene_p)}")
            if levene_p < 0.05:
                print("Transformation did not achieve homogeneous variances. Proceeding with Kruskal-Wallis.")
                best_transformation = None
        
        if not best_transformation:
            print("No successful transformation found. Switching to Kruskal-Wallis test.")
            try:
                kw_stat, kw_p = stats.kruskal(*groups)
                print(f"Kruskal-Wallis test p-value for {var}: {format_p_value(kw_p)}")
                if kw_p < 0.05:
                    print("Performing Dunn's post-hoc test...")
                    dunn_results = sp.posthoc_dunn(df, val_col=var, group_col='Damage', p_adjust='bonferroni')
                    print(f"Dunn's post-hoc test results for {var}:\n", dunn_results.applymap(format_p_value))
                    
                    # Compact Letter Display for Dunn's test
                    cld = generate_cld(dunn_results)
                    print(f"\nCompact Letter Display (CLD) for {var} - Dunn's test:")
                    for group, letter in sorted(cld.items()):
                        print(f"Damage Group {group}: {letter}")
            except Exception as e:
                print(f"Kruskal-Wallis or Dunn's test failed for {var}: {e}")
            continue




