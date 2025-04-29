import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ternary


def normalize_cations(df):
    total = df['Mg'] + df['Mn'] + df['Ca'] + df['Fe']
    df['Normalized_FeMn'] = (df['Fe'] + df['Mn']) / total
    df['Normalized_Mg'] = df['Mg'] / total
    df['Normalized_Ca'] = df['Ca'] / total
    return df


def plot_overlay_ternary(df1, df2, save_path="./temp/ternary_overlay.jpeg"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    data1 = df1[['Normalized_FeMn', 'Normalized_Mg', 'Normalized_Ca']].values
    data2 = df2[['Normalized_FeMn', 'Normalized_Mg', 'Normalized_Ca']].values

    figure, tax = ternary.figure(scale=1.0)
    tax.boundary(linewidth=1.5)
    tax.gridlines(color="black", multiple=0.1, linewidth=0.5)

    # Corner axis labels instead of edge labels
    tax.left_corner_label("Alm+Sps", fontsize=12)
    tax.right_corner_label("Pyrope", fontsize=12)
    tax.bottom_corner_label("Gross+Andr+Uva", fontsize=12)

    tax.set_title("Ternary Plot with Overlay", fontsize=16)

    # Plot both datasets with different colors
    tax.scatter(data1, marker='o', color='blue', label='Dataset 1',
                s=60, edgecolor='k', alpha=0.7)
    tax.scatter(data2, marker='^', color='red', label='Dataset 2',
                s=60, edgecolor='k', alpha=0.7)

    tax.legend(loc='upper right', fontsize=10)
    tax.ticks(axis='lbr', multiple=0.2, linewidth=1)
    tax.clear_matplotlib_ticks()
    tax._redraw_labels()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path
