import pandas as pd
import os
import json


csv_file_path = 'data/hcV3-stories.csv'
csv_df = pd.read_csv(csv_file_path)
csv_subset = csv_df[['summary']]

csv_subset['label'] = 1

# print(csv_subset[:3])
# exit()

json_folder_path = 'data/llama-2-70b-chat/outputs'
json_data_list = []

for filename in os.listdir(json_folder_path):
    if filename.endswith('.json'):
        json_file_path = os.path.join(json_folder_path, filename)
        with open(json_file_path, 'r') as json_file:
            # print('json_file_path')
            # print(json_file_path)
            data = json.load(json_file)
            # print(data)
            # print(type(data))
            # print(data['output']['choices'][0]['text'])
            extracted_data = [{'summary': data['output']['choices'][0]['text']}]
            json_data_list.extend(extracted_data)


json_df = pd.DataFrame(json_data_list)

json_df['label'] = 2

# print(json_df[:3])
# exit()
result_df = csv_subset.append(json_df)
# result_df = pd.concat([csv_subset, json_df], axis=1)


print(result_df)

result_df.to_csv('data/combined_70b.csv', index=False)
