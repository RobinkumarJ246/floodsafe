import sys
import time  # You can use this to simulate a long process if needed

# Check that correct arguments are being passed
if len(sys.argv) != 5:
    print("Error: Invalid number of arguments")
    sys.exit(1)

# Read and log the input values
elevation = sys.argv[1]
rainfall = sys.argv[2]
population_density = sys.argv[3]
impermeability_index = sys.argv[4]

print(f"Received data: Elevation = {elevation}, Rainfall = {rainfall}, Population Density = {population_density}, Impermeability Index = {impermeability_index}")

# Simulate processing time (optional)
time.sleep(3)  # Simulate some time-consuming operation

# Try converting values and handle potential errors
try:
    elevation = float(elevation)
    rainfall = float(rainfall)
    population_density = float(population_density)
    impermeability_index = float(impermeability_index)
    print("Data successfully converted to float.")
except ValueError as e:
    print(f"Error converting data: {e}")
    sys.exit(1)

# Perform prediction (this is a placeholder for actual logic)
print("Processing prediction...")
time.sleep(2)  # Simulating some computation time

# Returning dummy prediction result
print("Prediction result: Low Risk")  # Replace with actual prediction logic
sys.exit(0)