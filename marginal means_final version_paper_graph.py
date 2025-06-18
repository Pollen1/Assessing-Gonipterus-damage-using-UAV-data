# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 08:22:53 2024

@author: pnzuza
"""

#################################################################################################################"
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
file_path = 'E:/Final/Leave one stand approach/Ten band model final/All_site_data_te_indices.csv'  # Replace with the actual file path
df = pd.read_csv(file_path)
print(df.columns)  # Print the column names to ensure they are correct

# Define the columns used for the analysis
dependent_vars = ['MCARI','TCARI_OSAVI', 'LCI','NRI']
independent_vars = ['Damage', 'Site']

# Check if all dependent_vars exist in the dataframe
missing_vars = [var for var in dependent_vars if var not in df.columns]
if missing_vars:
    raise ValueError(f"Missing columns in the DataFrame: {missing_vars}")

# Convert categorical variables to category type if necessary
df[independent_vars] = df[independent_vars].astype('category')

# Function to calculate estimated marginal means
def calculate_emmeans(df, dependent_var, independent_vars):
    formula = f"{dependent_var} ~ {' + '.join(independent_vars)}"
    model = sm.formula.ols(formula, data=df).fit()
    
    # Predict marginal means for each combination of levels
    levels = df[independent_vars].drop_duplicates()
    predictions = levels.copy()
    predictions['emmeans'] = model.predict(levels)
    
    return predictions

# Calculate the estimated marginal means for each dependent variable
emmeans_results = {var: calculate_emmeans(df, var, independent_vars) for var in dependent_vars}

# Define the colors, labels, and marker styles for the plots
colors = sns.color_palette('bright', len(df['Site'].unique()))  # Use the bright color palette
labels = df['Site'].unique().tolist()
markers = ['o', 's', '^', 'x', 'd', '*', '+', 'p', 'h', 'H', 'D', 'v', '>', '<']  # Different marker styles

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(15, 12))

# Plot each variable in a subplot
for ax, (dependent_var, emmeans) in zip(axs.flatten(), emmeans_results.items()):
    for label, color, marker in zip(labels, colors, markers):
        subset = emmeans[emmeans['Site'] == label]
        ax.plot(subset['Damage'], subset['emmeans'], marker=marker, color=color, label=label, linestyle='-')
    ax.set_title(f"{dependent_var}", fontsize=15)
    ax.set_xlabel('Damage', fontsize=15)
    ax.set_ylabel(f"{dependent_var}", fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    
    # Add damage levels for all subplots
    damage_levels = ['No damage', 'Low', 'Medium', 'High']
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(damage_levels)

# Add subplot labels
subplot_labels = ['a', 'b', 'c', 'd']
for ax, label in zip(axs.flatten(), subplot_labels):
    ax.text(-0.1, 1.05, label, transform=ax.transAxes, fontsize=17, fontweight='bold', va='top', ha='right')

# Add a legend without boundary
fig.legend(labels, loc='center right', fontsize=18, frameon=False)

# Adjust layout
plt.tight_layout(rect=[0, 0, 0.75, 1])

# Save the plot
output_path = 'output_plot.png'  # Replace with the desired file path
plt.savefig(output_path, dpi=300)

# Show the plot
plt.show()


# Show the plot
plt.show()



