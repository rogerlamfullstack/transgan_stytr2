import pandas as pd
import glob
import os

# Path to your CSV files
csv_files = glob.glob("metrics/*.csv")

results = []

for f in csv_files:
    df = pd.read_csv(f)
    numeric_cols = ['loss_c', 'loss_s', 'ssim', 'psnr', 'lpips']
    
    # Compute average for each numeric column
    avg = df[numeric_cols].mean()
    avg['file'] = os.path.basename(f)
    
    results.append(avg)

# Combine into a DataFrame
df_avg = pd.DataFrame(results)[['file', 'loss_c', 'loss_s', 'ssim', 'psnr', 'lpips']]

# Print and save
print(df_avg)
df_avg.to_csv("average_per_csv.csv", index=False)
