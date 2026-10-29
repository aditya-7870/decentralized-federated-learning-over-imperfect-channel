import json
import matplotlib.pyplot as plt

# Load JSON data
with open("m1_all_results.json", "r") as file:
    data = json.load(file)

# Display the structure to identify the correct fields
print("JSON type:", type(data))

if isinstance(data, dict):
    print("Available keys:", list(data.keys()))

    for key, value in data.items():
        print(f"\n{key}:")
        print(str(value)[:500])

elif isinstance(data, list):
    print("Number of records:", len(data))

    if data:
        print("First record:", data[0])

# No graph is generated yet because the data fields
# must be identified before plotting.