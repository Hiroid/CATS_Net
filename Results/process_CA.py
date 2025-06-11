import pandas as pd

# Read the original CSV file
input_file = './CA_fishZ_rho_Control_long_format.csv'
output_file = './CA_results.csv'

try:
    df = pd.read_csv(input_file)

    # Group by Layer, ModelNumber, ROIName and calculate the average of Correlation and PartialCorrelation
    # Use .agg() method to calculate mean and reset index
    averaged_df = df.groupby(['Layer', 'ModelNumber', 'ROIName'])[['Correlation', 'PartialCorrelation']].mean().reset_index()


    # Write results to new CSV file without index
    averaged_df.to_csv(output_file, index=False)

    print(f"Processing completed, results saved to {output_file}")

except FileNotFoundError:
    print(f"Error: Input file {input_file} not found.")
except KeyError as e:
    print(f"Error: Missing column in input file: {e}. Please ensure the file contains 'Layer', 'ModelNumber', 'ROIName', 'Correlation', 'PartialCorrelation' columns.")
except Exception as e:
    print(f"Error occurred during processing: {e}")
