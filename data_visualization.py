This script provides a method for analyzing and visualizing the distribution of symptoms

import numpy as np
from matplotlib import pyplot as plt

def data_analysis(df, column_name, color='lightblue', text_height=0.02, text_width=0.05):
    counts = df[column_name].value_counts().values 
    cls_names = df[column_name].value_counts().keys()
    width, text_width, text_height = 0.3, text_width, 0.8
    fig, ax = plt.subplots(figsize=(15, 5))
    indices = np.arange(len(counts))
    ax.bar(indices, counts, width, color=color)
    ax.set_xlabel("Symptoms", color="orange")
    ax.set_xticklabels(cls_names, rotation=60)
    ax.set(xticks=indices, xticklabels=cls_names)
    ax.set_ylabel('Frequency', color='grey')

    for i, v in enumerate(counts):  
        ax.text(i - text_width, v + text_height, str(v), color="green")


data_analysis(df=data, column_name="itching", color="lightblue")
