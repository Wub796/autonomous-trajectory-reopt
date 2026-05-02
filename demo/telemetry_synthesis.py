import pandas as pd
import numpy as np

# 1. Define the parameters of our simulated mission
total_hours = 2400
anomaly_start = 1500

# 2. Generate the Time column (Hours 1 to 2400)
# np.arange creates a sequence of numbers
time_array = np.arange(1, total_hours + 1)

# 3. Generate the Baseline Telemetry (Healthy Spacecraft)
# np.random.normal(mean_value, standard_deviation, total_number_of_items)
# We simulate the thruster operating at 99% with a tiny 0.5% fluctuation
thruster_data = np.random.normal(99.0, 0.5, total_hours)

# 4. INJECT THE ANOMALY (YOUR TURN)
# How do you tell Python to take the 'thruster_data' array, 
# look ONLY at the items from index 1500 to the end, 
# and lower their values to simulate the hardware failure?

# [WRITE YOUR LOGIC HERE]
thruster_data[1500:] -= 14.0

solar_temp_data = np.random.normal(45.0, 0.5, total_hours)

# 5. Construct the DataFrame (The Spreadsheet)
mission_dataframe = pd.DataFrame({
    'Hour': time_array,
    'Thruster_Efficiency': thruster_data,
    'Solar_Array_Temp': solar_temp_data
})

# 6. Save the data to a CSV file
mission_dataframe.to_csv('simulated_telemetry.csv', index=False)
print("Synthetic telemetry dataset generated successfully.")