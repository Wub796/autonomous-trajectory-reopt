import pandas as pd
from sklearn.ensemble import IsolationForest

# 1. LOAD THE DATA (YOUR TURN)
# You used mission_dataframe.to_csv() to save the file earlier.
# What is the pandas command to READ a csv file named 'simulated_telemetry.csv' 
# and save it as a variable named 'flight_data'?

# [WRITE YOUR LOGIC HERE]
flight_data = pd.read_csv("simulated_telemetry.csv")

# 2. Isolate the Features (The sensors the AI will monitor)
# We don't want the AI looking at the 'Hour' column, only the physical sensors.
sensor_data = flight_data[['Thruster_Efficiency', 'Solar_Array_Temp']]

# 3. Initialize the AI Model
# contamination=0.05 tells the AI we expect about 5% of our mission data to be anomalous
ai_model = IsolationForest(contamination=0.05, random_state=42)

# 4. Train the AI and make it predict failures
print("Training AI on healthy flight telemetry...")
healthy_baseline = sensor_data[0:500]
ai_model.fit(healthy_baseline)

# 5. Record the AI's findings back into our spreadsheet
flight_data['AI_Status_Flag'] = ai_model.predict(sensor_data)

# Print the exact hours where the AI detected a failure (where flag equals -1)
failures_detected = flight_data[flight_data['AI_Status_Flag'] == -1]

# Print the TOTAL number of anomalies the AI found in the 2400 hour mission
print(f"Total anomalies detected: {len(failures_detected)}")

# Filter the list to only show failures that happened AFTER hour 1495
catastrophic_failures = failures_detected[failures_detected['Hour'] > 1495]
print("\nAnomalies detected in the degradation zone:")
print(catastrophic_failures['Hour'].head(10))