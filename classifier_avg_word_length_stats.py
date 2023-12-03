import pandas as pd

def read_and_print_min_max_with_filter(file_path, column_name, filter_column, filter_value):
    # Read CSV file into a Pandas DataFrame
    df = pd.read_csv(file_path)

    # Apply filter on the specified column
    filtered_df = df[df[filter_column] == filter_value]

    # Print the filtered DataFrame
    # print(f"\nFiltered DataFrame from file '{file_path}' where '{filter_column}' is '{filter_value}':")
    # print(filtered_df)

    # Check if there are any rows after applying the filter
    if not filtered_df.empty:
        # Calculate and print the min and max values of the specified column
        min_value = filtered_df[column_name].min()
        max_value = filtered_df[column_name].max()
        print(f"\nMinimum value in column '{column_name}': {min_value}")
        print(f"Maximum value in column '{column_name}': {max_value}")
        if max_value > 200:
            print(f"Max value is greater than 200: \n{max_value}")
    else:
        print(f"\nNo rows after applying the filter.")

# Specify the file paths for your CSV files
file_path1 = './data/train_stratify_avg_word_length.csv'
file_path2 = './data/test_stratify_avg_word_length.csv'

# Read and print min/max values with the filter for the first file
read_and_print_min_max_with_filter(file_path1, 'avg_word_length', 'label', 1)
read_and_print_min_max_with_filter(file_path1, 'avg_word_length', 'label', 2)

# Read and print min/max values with the filter for the second file
read_and_print_min_max_with_filter(file_path2, 'avg_word_length', 'label', 1)
read_and_print_min_max_with_filter(file_path2, 'avg_word_length', 'label', 2)
