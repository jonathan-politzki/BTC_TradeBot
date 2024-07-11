import matplotlib.pyplot as plt
import pandas as pd  # This is necessary if you're manipulating DataFrames in this script or elsewhere before plotting
import seaborn as sb

def plot_data(df, title):
    plt.figure(figsize=(15, 5))
    plt.plot(df['date'], df['close'])
    plt.title(title, fontsize=15)
    plt.ylabel('Price in Dollars ($)')
    plt.xlabel('Date')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility if needed
    plt.grid(True)
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.show()

# correlation mapping shows us that the date and OHLC are super correlated obviously. Test this later.

def correlation_mapping_filtered():
    sb.heatmap(filtered_df.corr() > 0.9, annot=True, cbar=False)
    plt.show()


def correlation_mapping_new():
    sb.heatmap(updated_df.corr() > 0.9, annot=True, cbar=False)
    plt.show()