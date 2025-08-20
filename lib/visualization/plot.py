import matplotlib.pyplot as plt

def plot_dataframe(df, title="DataFrame Plot", xlabel="Index", ylabel="Values"):
    """
    Plots all columns of a pandas DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame to plot.
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
    """
    ax = df.plot(figsize=(10, 6))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()