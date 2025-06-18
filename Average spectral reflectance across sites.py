# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 05:07:55 2025

@author: pnzuza
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from string import ascii_lowercase

# List of CSV files containing spectral data
csv_files = ['C:/Users/pnzuza/Downloads/spectral reflectance/Feb 2022_Hodgsons.csv',
             'C:/Users/pnzuza/Downloads/spectral reflectance/May 2022 Hodgsons.csv',
             'C:/Users/pnzuza/Downloads/spectral reflectance/Feb 2022 Sutton.csv',
             'C:/Users/pnzuza/Downloads/spectral reflectance/April 2022 Sutton.csv',
             'C:/Users/pnzuza/Downloads/spectral reflectance/May 2022 Sutton.csv',
             #'C:/Users/pnzuza/Downloads/spectral reflectance/5band/August 2022 Sutton.csv',
             #'C:/Users/pnzuza/Downloads/spectral reflectance/5band/October 2022 Mooiplass.csv',
             #'C:/Users/pnzuza/Downloads/spectral reflectance/5band/October 2022 Mooiplass f2.csv',
             #'C:/Users/pnzuza/Downloads/spectral reflectance/5band/November Mooiplass.csv',
             #'C:/Users/pnzuza/Downloads/spectral reflectance/5band/Melmoth_November_f1.csv',
             'C:/Users/pnzuza/Downloads/spectral reflectance/Jan 2023_Mooiplass.csv',
             'C:/Users/pnzuza/Downloads/spectral reflectance/Piet retief_stand1.csv',
             'C:/Users/pnzuza/Downloads/spectral reflectance/Piet Retief stand2.csv',
             'C:/Users/pnzuza/Downloads/spectral reflectance/Piet retief stand3.csv'
            ]

figure_headings = [
    'Gre_Feb22',
    'Gre_May22',
    'Ixo_Feb22',
    'Ixo_Apr22',
    'Ixo_May22',
    #'Ixopo Stand 1 Aug',
    #'Melmoth Stand 1 Oct',
    #'Melmoth Stand 1 Nov',
    #'Melmoth Stand 2 Oct',
    'Mel2_Nov22',
    'Mel2_Jan23',
    'PR1_Mar23',
    'PR2 Mar23',
    'PR3_Mar23'
]

# Calculate the number of rows and columns for the subplots
num_files = len(csv_files)
num_rows = (num_files + 2) // 3  # Adjust the number of rows as needed
num_cols = 3

# Mapping of damage level to marker and color
marker_dict = {'No damage': 'o', 'Low': 's', 'Medium': '^', 'High': 'd'}
color_dict = {
    'No damage': 'blue',
    'Low': 'green',
    'Medium': 'orange',
    'Medium ': 'orange',  # Typo: 'Medium' is spelled incorrectly here
    'High': 'red',
    'Low ': 'green',
    'No damage ': 'blue'
}

# Initialize an empty set to store unique damage labels
unique_labels = set()

# Create subplots
fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(55, 20 * num_rows))
axes = axes.flatten()

for i, csv_file in enumerate(csv_files):
    if i < num_files:
        # Load data from the CSV file
        data = pd.read_csv(csv_file)

        # Plotting
        plt.sca(axes[i])
        plt.subplots_adjust(wspace=0.3, hspace=0.5)  # Adjust subplot spacing

        data_melted = data.melt(id_vars=['Damage'], var_name='Wavelength', value_name='Reflectance')
        data_melted['Wavelength'] = pd.to_numeric(data_melted['Wavelength'], errors='coerce')

        sns.set(style='white')
        sns.lineplot(data=data_melted, x='Wavelength', y='Reflectance', hue='Damage', style='Damage',
                     markers=True, err_style='band', palette={'No damage': 'blue', 'Low': 'green', 'Medium': 'orange', 'High': 'red'},
                     linewidth=15, legend=False)

        for spine in plt.gca().spines.values():
            spine.set_linewidth(4)

        plt.xlabel('Wavelength nm', fontsize=60)
        plt.ylabel('Reflectance %', fontsize=60)

        plt.tick_params(axis='x', labelsize=55, pad=50)
        plt.tick_params(axis='y', labelsize=55, pad=50)
        plt.ylim(0, 0.6)
        plt.xlim(400, 800)

        label = ascii_lowercase[i]
        plt.text(0.05, 0.9, f'({label}) {figure_headings[i]}', transform=plt.gca().transAxes, fontsize=60)

        # Gather unique labels for legend creation
        unique_labels.update(data['Damage'].unique())

# Create a combined legend for all subplots outside the figure
legend_elements = [plt.Line2D([0], [0], marker=marker_dict[label], color='w', label=label, markerfacecolor=color_dict[label], markersize=30)
                   for label in marker_dict.keys()]

# Show legend with annotations in black color and increased width
# Show legend with annotations in black color and increased width
legend = plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.3), fontsize=65, labelspacing=2, markerscale=2, ncol=4,frameon=False)  # ncol parameter sets the number of columns
plt.setp(legend.get_texts(), fontsize='65')  # Adjust text properties
plt.subplots_adjust(bottom=0.15)  # Adjust layout to accommodate the legend at the bottom

for i in range(num_files, num_rows * num_cols):
    fig.delaxes(axes[i])

# Remove the border around the figure
sns.despine()
#plt.tight_layout(rect=[0, 0.12, 1, 0.88])

#plt.savefig('D:/figure/spectral.jpeg',dpi=300, bbox_inches='tight') 

# Show the plot
plt.show()





import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from string import ascii_lowercase

# List of CSV files containing spectral data
csv_files = [
    'C:/Users/pnzuza/Downloads/spectral reflectance/Feb 2022_Hodgsons.csv',
    'C:/Users/pnzuza/Downloads/spectral reflectance/May 2022 Hodgsons.csv',
    'C:/Users/pnzuza/Downloads/spectral reflectance/Feb 2022 Sutton.csv',
    'C:/Users/pnzuza/Downloads/spectral reflectance/April 2022 Sutton.csv',
    'C:/Users/pnzuza/Downloads/spectral reflectance/May 2022 Sutton.csv',
    #'C:/Users/pnzuza/Downloads/spectral reflectance/5band/November Mooiplass.csv',
    'C:/Users/pnzuza/Downloads/spectral reflectance/Jan 2023_Mooiplass.csv',
    'C:/Users/pnzuza/Downloads/spectral reflectance/Piet retief_stand1.csv',
    'C:/Users/pnzuza/Downloads/spectral reflectance/Piet Retief stand2.csv',
    'C:/Users/pnzuza/Downloads/spectral reflectance/Piet retief stand3.csv'
]

figure_headings = [
    'Gre_Feb22', 'Gre_May22', 'Ixo_Feb22', 'Ixo_Apr22', 'Ixo_May22',
    'Mel2_Jan23', 'PR1_Mar23', 'PR2 Mar23', 'PR3_Mar23'
]

# Plot layout
num_files = len(csv_files)
num_cols = 3
num_rows = (num_files + num_cols - 1) // num_cols

# Define markers and hex colors
marker_dict = {'No damage': 'o', 'Low': 's', 'Medium': '^', 'High': 'd'}
color_dict = {
    'No damage': '#897ff6',
    'Low': '#8af482',
    'Medium': '#ffba75',
    'High': '#ff7c7c'
}

# Start plotting
fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(55, 20 * num_rows))
axes = axes.flatten()

for i, csv_file in enumerate(csv_files):
    if i < num_files:
        data = pd.read_csv(csv_file)
        data['Damage'] = data['Damage'].str.strip()  # Clean labels

        plt.sca(axes[i])
        plt.subplots_adjust(wspace=0.3, hspace=0.5)

        data_melted = data.melt(id_vars=['Damage'], var_name='Wavelength', value_name='Reflectance')
        data_melted['Wavelength'] = pd.to_numeric(data_melted['Wavelength'], errors='coerce')

        sns.set(style='white')
        sns.lineplot(
            data=data_melted,
            x='Wavelength', y='Reflectance',
            hue='Damage', style='Damage',
            markers=True,
            err_style='band',
            palette=color_dict,
            linewidth=15,
            legend=False
        )

        for spine in plt.gca().spines.values():
            spine.set_linewidth(4)

        plt.xlabel('Wavelength nm', fontsize=60)
        plt.ylabel('Reflectance %', fontsize=60)
        plt.tick_params(axis='x', labelsize=55, pad=50)
        plt.tick_params(axis='y', labelsize=55, pad=50)
        plt.ylim(0, 0.6)
        plt.xlim(400, 800)

        label = ascii_lowercase[i]
        plt.text(0.05, 0.9, f'({label}) {figure_headings[i]}', transform=plt.gca().transAxes, fontsize=60)

# Legend outside plot
legend_elements = [
    plt.Line2D([0], [0], marker=marker_dict[label], color='w',
               label=label, markerfacecolor=color_dict[label], markersize=30)
    for label in ['No damage', 'Low', 'Medium', 'High']
]

legend = plt.legend(
    handles=legend_elements,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.3),
    fontsize=65,
    labelspacing=2,
    markerscale=2,
    ncol=4,
    frameon=False
)
plt.setp(legend.get_texts(), fontsize='65')
plt.subplots_adjust(bottom=0.15)

# Remove extra subplots if any
for i in range(num_files, num_rows * num_cols):
    fig.delaxes(axes[i])

sns.despine()

# Save or show plot
# plt.savefig('D:/figure/spectral.jpeg', dpi=300, bbox_inches='tight')
plt.show()


































