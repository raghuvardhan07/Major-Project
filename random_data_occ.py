import csv
import random
import os

def generate_occupancy_data():
    """
    Generates random occupancy data ranging from 0 to 75.

    The distribution is weighted such that values between 35 and 60 are most common,
    with specific outliers like 10 and 74 also possible, and other values
    occurring less frequently.
    """
    population = []
    weights = []

    # Main range (35-60): Highest probability
    main_range = list(range(35, 61))
    population.extend(main_range)
    weights.extend([10] * len(main_range)) # High weight

    # Specific outlier 10: Moderate probability
    population.append(10)
    weights.append(5) # Moderate weight

    # Specific outlier 74: Moderate probability
    population.append(74)
    weights.append(5) # Moderate weight

    # Lower values (0-9, 11-34): Low probability
    lower_range = list(range(0, 10)) + list(range(11, 35))
    population.extend(lower_range)
    weights.extend([1] * len(lower_range)) # Low weight

    # Higher values (61-73, 75): Low probability
    higher_range = list(range(61, 74)) + [75]
    population.extend(higher_range)
    weights.extend([1] * len(higher_range)) # Low weight

    # Randomly select one value based on the defined populations and weights
    return random.choices(population, weights, k=1)[0]

# Define the input/output file path
file_path = 'harmonized_dataset.csv'
temp_file_path = 'harmonized_dataset_temp.csv'

# Check if the input file exists
if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
else:
    try:
        # Read the original data
        with open(file_path, mode='r', newline='') as infile:
            reader = csv.reader(infile)
            header = next(reader) # Read the header row
            data = list(reader)   # Read the rest of the data

        # Find the index of the 'occupancy' column
        try:
            occupancy_col_index = header.index('occupancy')
        except ValueError:
            print("Error: 'occupancy' column not found in the header.")
            exit() # Exit if the column doesn't exist

        # Modify the data with new random occupancy values
        for row in data:
            # Ensure the row has enough columns before trying to access the index
            if len(row) > occupancy_col_index:
                 # Generate random occupancy and ensure it's stored as a string for CSV writing
                row[occupancy_col_index] = str(generate_occupancy_data())
            else:
                # Handle rows that might be shorter than expected (e.g., add padding or log warning)
                print(f"Warning: Row {reader.line_num} is shorter than expected and cannot be processed.")
                # Optionally add empty strings or default values if appropriate
                while len(row) <= occupancy_col_index:
                     row.append('') # Append empty strings to match header length
                # Now set the occupancy value after ensuring the column exists
                row[occupancy_col_index] = str(generate_occupancy_data())


        # Write the modified data to a temporary file first
        with open(temp_file_path, mode='w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(header) # Write the header
            writer.writerows(data)  # Write the modified data rows

        # Replace the original file with the temporary file
        os.replace(temp_file_path, file_path)

        print(f"Successfully updated 'occupancy' data in {file_path}")

    except Exception as e:
        print(f"An error occurred: {e}")
        # Clean up the temporary file if it exists and an error occurred
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path) 