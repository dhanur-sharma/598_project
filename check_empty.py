import os
import json

folder_path = '/common/home/ds1987/598_project/598_project/outputs'

empty_field_count = 0

for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as file:
            data = json.load(file)
            if data['output']['choices'][0]['text'] == '' or data['output']['choices'][0]['text'] == '/n/n':
                empty_field_count += 1

print(f"Number of files with empty field or '/n/n': {empty_field_count}")
