import csv
import json

def merge_csv_to_json(json_file, population_csv, elevation_csv, output_json):
    # Load the JSON data
    with open(json_file, 'r') as j_file:
        json_data = json.load(j_file)
    
    # Load the population data
    population_data = {}
    with open(population_csv, 'r') as pop_file:
        pop_reader = csv.DictReader(pop_file)
        for row in pop_reader:
            ward = int(row['ward'])
            population_data[ward] = int(row['population'])
    
    # Load the elevation data
    elevation_data = {}
    with open(elevation_csv, 'r') as elev_file:
        elev_reader = csv.DictReader(elev_file)
        for row in elev_reader:
            ward = int(row['ward'])
            elevation_data[ward] = float(row['avg_elev'])
    
    # Merge data into JSON
    for item in json_data:
        ward = item['ward']
        if ward in population_data:
            item['population'] = population_data[ward]
        if ward in elevation_data:
            item['avg_elev'] = elevation_data[ward]
    
    # Write updated JSON to output file
    with open(output_json, 'w') as out_file:
        json.dump(json_data, out_file, indent=4)
    
    print(f"Merged data written to {output_json}")

# File paths
json_file = 'impermeability.json'  # Replace with the path to your JSON file
population_csv = 'chennai_data_2011.csv'  # Replace with the path to your population CSV file
elevation_csv = 'avg_elev.csv'  # Replace with the path to your elevation CSV file
output_json = 'wardwise_data.json'  # Replace with the path for the output JSON file

# Run the function
merge_csv_to_json(json_file, population_csv, elevation_csv, output_json)