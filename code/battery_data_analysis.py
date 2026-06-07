# Import dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.signal import savgol_filter
import sys
import os
# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))
# Now import the modules
from utils.data_loader import load_nasa_battery_data
from utils.plotting import plot_capacity_degradation

# Load battery data
data_dir = '../data/NASA_batteries/'
batteries = load_nasa_battery_data(data_dir)

# Display basic information
print(f"Loaded data for {len(batteries)} batteries")
battery_names = list(batteries.keys())
print(f"Battery IDs: {battery_names}")

# Examine structure of first battery
first_battery = batteries[battery_names[0]]
first_battery['summary'].head()

def preprocess_battery_data(batteries):
    """Clean and preprocess battery data"""
    processed_data = {}
    
    for battery_id, battery_data in batteries.items():
        # Get summary data
        summary = battery_data['summary'].copy()
        
        # Filter out invalid cycles
        summary = summary[summary['Capacity (Ah)'] > 0]
        
        # Calculate SOH as percentage of initial capacity
        initial_capacity = summary['Capacity (Ah)'].iloc[0]
        summary['SOH'] = summary['Capacity (Ah)'] / initial_capacity
        
        # Add to processed data
        processed_data[battery_id] = {
            'summary': summary,
            'cycles': battery_data['cycles']
        }
    
    return processed_data

# Preprocess data
processed_batteries = preprocess_battery_data(batteries)

# Plot capacity degradation for all batteries
plt.figure(figsize=(10, 6))
for battery_id, battery_data in processed_batteries.items():
    plt.plot(battery_data['summary']['Cycle'], 
             battery_data['summary']['Capacity (Ah)'], 
             label=battery_id)
    
plt.xlabel('Cycle Number')
plt.ylabel('Capacity (Ah)')
plt.title('Capacity Degradation Over Cycles')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(data_dir, 'capacity_degradation.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot SOH degradation
plt.figure(figsize=(10, 6))
for battery_id, battery_data in processed_batteries.items():
    plt.plot(battery_data['summary']['Cycle'], 
             battery_data['summary']['SOH'], 
             label=battery_id)
    
plt.xlabel('Cycle Number')
plt.ylabel('State of Health')
plt.title('SOH Degradation Over Cycles')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(data_dir, 'SOH_degradation.png'), dpi=300, bbox_inches='tight')
plt.close()

# Analyze voltage and current profiles for a single cycle
battery_id = battery_names[0]
cycle_number = 10  # Choose an early cycle
cycle_data = processed_batteries[battery_id]['cycles'][cycle_number]

# Debug: Check what keys are available in cycle_data
print(f"Available keys in cycle_data: {cycle_data.keys()}")

# Plot available data for this cycle (either charge, discharge, or both)
plt.figure(figsize=(12, 8))

if 'discharge' in cycle_data:
    # Plot discharge data
    plt.subplot(2, 1, 1)
    plt.plot(cycle_data['discharge']['Time (s)'], cycle_data['discharge']['Voltage (V)'], label='Voltage')
    plt.ylabel('Voltage (V)')
    plt.title(f'Discharge Profile - Battery {battery_id}, Cycle {cycle_number}')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(cycle_data['discharge']['Time (s)'], cycle_data['discharge']['Current (A)'], label='Current')
    plt.xlabel('Time (s)')
    plt.ylabel('Current (A)')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, 'discharge.png'), dpi=300, bbox_inches='tight')
    plt.close()
else:
    print(f"No discharge data found for cycle {cycle_number}")

# If you want to find a cycle with charge data, you can use this:
print("\nSearching for a cycle with charge data...")
for cycle_num in sorted(processed_batteries[battery_id]['cycles'].keys()):
    if 'charge' in processed_batteries[battery_id]['cycles'][cycle_num]:
        print(f"Found cycle with charge data: {cycle_num}")
        
        # Plot charge data for this cycle
        charge_cycle_data = processed_batteries[battery_id]['cycles'][cycle_num]
        
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(charge_cycle_data['charge']['Time (s)'], charge_cycle_data['charge']['Voltage (V)'], label='Voltage')
        plt.ylabel('Voltage (V)')
        plt.title(f'Charge Profile - Battery {battery_id}, Cycle {cycle_num}')
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(charge_cycle_data['charge']['Time (s)'], charge_cycle_data['charge']['Current (A)'], label='Current')
        plt.xlabel('Time (s)')
        plt.ylabel('Current (A)')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(data_dir, 'charge.png'), dpi=300, bbox_inches='tight')
        plt.close()

        break
else:
    print("No cycles with charge data found")

def extract_features(batteries):
    """Extract features from battery cycles"""
    features_list = []
    total_flagged = 0
    for battery_id, battery_data in batteries.items():
        summary = battery_data['summary']
        cycles = battery_data['cycles']
        previous_rows = []
        max_previous_rows = 10
        
        for idx, row in summary.iterrows():
            cycle_num = int(row['Cycle'])
            flagged = False
            
            if cycle_num not in cycles:
                continue
            
            cycle_data = cycles[cycle_num]
            
            # Initialize feature dictionary with common fields
            features = {
                'battery_id': battery_id,
                'cycle': cycle_num,
                'capacity': row['Capacity (Ah)'],
                'soh': row['SOH'],
            }
            
            # Add charge features if available
            if 'charge' in cycle_data:
                charge = cycle_data['charge']
                
                features.update({
                    'charge_time': charge['Time (s)'].max(),
                    'charge_voltage_mean': charge['Voltage (V)'].mean(),
                    'charge_voltage_max': charge['Voltage (V)'].max(),
                    'charge_current_mean': charge['Current (A)'].mean(),
                    
                    # Temperature features (if available)
                    'charge_temp_mean': charge['Temperature (C)'].mean() if 'Temperature (C)' in charge else np.nan,
                    
                    # Energy feature
                    'charge_energy': np.trapz(charge['Voltage (V)'] * charge['Current (A)'], charge['Time (s)'])
                })

            # Add discharge features if available
            if 'discharge' in cycle_data:
                discharge = cycle_data['discharge']
                
                features.update({
                    'discharge_time': discharge['Time (s)'].max(),
                    'discharge_voltage_mean': discharge['Voltage (V)'].mean(),
                    'discharge_voltage_min': discharge['Voltage (V)'].min(),
                    'discharge_current_mean': abs(discharge['Current (A)'].mean()),
                    
                    # Temperature features (if available)
                    'discharge_temp_mean': discharge['Temperature (C)'].mean() if 'Temperature (C)' in discharge else np.nan,
                    
                    # Energy feature
                    'discharge_energy': abs(np.trapz(discharge['Voltage (V)'] * discharge['Current (A)'], discharge['Time (s)']))
                })
            
            # Add features if available
            for previous_index in range(1, max_previous_rows + 1):
                str_previous_index = str(previous_index)
                if len(str_previous_index) == 1:
                    str_previous_index = '0' + str_previous_index
                previous_row = row
                if previous_index <= len(previous_rows):
                    previous_row = previous_rows[-previous_index]
                else:
                    flagged = True
                    if len(previous_rows):
                        previous_row = previous_rows[0]

                features.update({
                    'soh_' + str_previous_index: previous_row['SOH']
                })

                cycle_num_prev = int(previous_row['Cycle'])
                
                if cycle_num_prev not in cycles:
                    continue
                
                cycle_data_prev = cycles[cycle_num_prev]
            
                # Add charge features if available
                if 'charge' in cycle_data_prev:
                    charge_prev = cycle_data_prev['charge']
                    
                    features.update({
                        'charge_time_' + str_previous_index: charge_prev['Time (s)'].max(),
                        'charge_voltage_mean_' + str_previous_index: charge_prev['Voltage (V)'].mean(),
                        'charge_voltage_max_' + str_previous_index: charge_prev['Voltage (V)'].max(),
                        'charge_current_mean_' + str_previous_index: charge_prev['Current (A)'].mean(),
                        
                        # Temperature features (if available)
                        'charge_temp_mean_' + str_previous_index: charge_prev['Temperature (C)'].mean() if 'Temperature (C)' in charge else np.nan,
                        
                        # Energy feature
                        'charge_energy_' + str_previous_index: np.trapz(charge_prev['Voltage (V)'] * charge_prev['Current (A)'], charge_prev['Time (s)'])
                    })

                # Add discharge features if available
                if 'discharge' in cycle_data_prev:
                    discharge_prev = cycle_data_prev['discharge']
                    
                    features.update({
                        'discharge_time_' + str_previous_index: discharge_prev['Time (s)'].max(),
                        'discharge_voltage_mean_' + str_previous_index: discharge_prev['Voltage (V)'].mean(),
                        'discharge_voltage_min_' + str_previous_index: discharge_prev['Voltage (V)'].min(),
                        'discharge_current_mean_' + str_previous_index: abs(discharge_prev['Current (A)'].mean()),
                        
                        # Temperature features (if available)
                        'discharge_temp_mean_' + str_previous_index: discharge_prev['Temperature (C)'].mean() if 'Temperature (C)' in discharge else np.nan,
                        
                        # Energy feature
                        'discharge_energy_' + str_previous_index: abs(np.trapz(discharge_prev['Voltage (V)'] * discharge_prev['Current (A)'], discharge_prev['Time (s)']))
                    })

            # Only add to features_list if we have at least one of charge or discharge data
            if 'charge' in cycle_data or 'discharge' in cycle_data:
                if not flagged:
                    features_list.append(features)
                else:
                    total_flagged += 1
                previous_rows.append(row)
                previous_rows = previous_rows[-max_previous_rows:]
    print("Flagged:", total_flagged)
    # Create DataFrame
    features_df = pd.DataFrame(features_list)
    
    return features_df

# Extract features
features_df = extract_features(processed_batteries)

# Debug: Check available columns
print("Available columns in features_df:")
print(features_df.columns.tolist())

# Check how many cycles have both charge and discharge data
if 'charge_energy' in features_df.columns and 'discharge_energy' in features_df.columns:
    both_count = features_df.dropna(subset=['charge_energy', 'discharge_energy']).shape[0]
    print(f"\nCycles with both charge and discharge data: {both_count} out of {features_df.shape[0]}")

# Check feature correlation with SOH
plt.figure(figsize=(12, 10))
numeric_features = features_df.select_dtypes(include=['float64', 'int64']).drop(['soh', 'capacity'], axis=1)
correlation = numeric_features.corrwith(features_df['soh']).sort_values(ascending=False)

sns.barplot(x=correlation.values, y=correlation.index)
plt.title('Feature Correlation with State of Health')
plt.xlabel('Correlation Coefficient')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(data_dir, 'correlation.png'), dpi=300, bbox_inches='tight')
plt.close()

# Select important features
important_features = sorted([c for c in correlation[abs(correlation) > 0.5].index.tolist() if c != "cycle"])
not_important_features = sorted([c for c in correlation[abs(correlation) <= 0.5].index.tolist() if c != "cycle"])

print("Selected features:", len(important_features), important_features)
print("Not selected features:", len(not_important_features), not_important_features)

# Prepare data
X = features_df[important_features]
y_soh = features_df['soh']
y_capacity = features_df['capacity']

# Save processed data for model training
import pickle
os.makedirs('../models', exist_ok=True)

with open('../models/data.pkl', 'wb') as f:
    pickle.dump({
        'X': X,
        'y_soh': y_soh,
        'y_capacity': y_capacity,
        'features': important_features,
        'metadata': features_df[['battery_id', 'cycle']]
    }, f)

print("Data processing complete. Files saved to ../models/")