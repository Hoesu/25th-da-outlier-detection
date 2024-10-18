import matplotlib.pyplot as plt
import os

def displayPlot(df, title='Anomaly Detection', saveName=None):
    """
    Arguments:
    df (DataFrame)
    : The input DataFrame that must contain the following columns:
        - 'ds': A timestamp or datetime column representing the x-axis.
        - 'y': A numeric column representing the y-axis values.
        - 'anomaly': A boolean column (True/False) indicating whether a point is an anomaly.

    title (str, optional)
    : The title of the plot. Default is 'Anomaly Detection'.

    saveName (str, optional)
    : The file name (including extension like .png) to save the plot. 
    If not provided, the plot is displayed without being saved.
    """

    plt.figure(figsize=(12, 6))

    # normal data
    plt.scatter(df[df['anomaly'] == False]['ds'], df[df['anomaly'] == False]['y'], color='black', label='Normal', s=1)
    # anomaly data
    plt.scatter(df[df['anomaly'] == True]['ds'], df[df['anomaly'] == True]['y'], color='red', label='Anomaly', s=1)

    plt.plot(df['ds'], df['y'], color='gray', alpha=0.5)

    plt.xlabel('Date Time')
    plt.ylabel('y Value')
    plt.title(title)
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid()

    plt.tight_layout()

    if saveName is not None:
        plt.savefig(saveName) 
    
    plt.show()