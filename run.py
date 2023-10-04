import together
import pandas as pd
import time
import json
from pprint import pprint

import config

start_time = time.time()
def read_file(file_path):
    return pd.read_csv(file_path)

together.api_key = "8459dc8088636ae70d85686b1e180015de1828643fcca4ae3c39567e5d640852"

# start the vm for the model
together.Models.start(config.MODEL)

df = read_file(config.FILE_PATH)

stories = df['story']
stories = stories[:config.NUMBER_OF_STORIES]
print(stories)
# print(len(stories))
# exit()
outputs = []

instruction = config.INSTRUCTION
for i, story in enumerate(stories):
    print(story)
    prompt = instruction + str(story)
    summary_output = together.Complete.create(
                        prompt = prompt, 
                        model = config.MODEL, 
                        max_tokens = 256,
                        temperature = 0.7, # default value
                        top_k = 50, # default value
                        top_p = 0.7, # default value
                        repetition_penalty = 1, # default value
                        stop = ['<human>', '\n\n']
                        )
    json_object = json.dumps(outputs, indent=4)
    outfile = open(f'./outputs/output_{i}.json', 'w+')
    outfile.write(json_object)
    outfile.close()
    outputs.append(summary_output)

print('*'*100)
pprint(outputs)

# save outputs
# json_object = json.dumps(outputs, indent=4)

# with open("/output/outputs.json", "w") as outfile:
#     outfile.write(json_object)

output_strings = []
for out in outputs:
    output_strings.append(out['output']['choices'][0]['text'])

print('*'*100)
print(output_strings)
end_time = time.time()

print('Time taken:', end_time - start_time)

# print generated text
# print(output['prompt'][0]+output['output']['choices'][0]['text'])

# stop
# together.Models.stop("togethercomputer/llama-2-7b-chat")
together.Models.stop(config.MODEL)

# check which models have started or stopped
together.Models.instances()