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

# pprint(together.Models.list())
# with open('temp/models.json', 'w+') as f:
#     json.dump(together.Models.list(), f, indent=4)
# exit()
# start the vm for the model
together.Models.start(config.MODEL)
print(f'Model {config.MODEL} started.')

df = read_file(config.FILE_PATH)

stories = df['story']
stories = stories[config.STORY_START:config.NUMBER_OF_STORIES]
# print(stories)
# print(len(stories))
# exit()
outputs = []
i = config.STORY_START
instruction = config.INSTRUCTION
for story in stories:
    # print(story)
    prompt = instruction + str(story)
    summary_output = together.Complete.create(
                        prompt = prompt, 
                        model = config.MODEL, 
                        max_tokens = config.MAX_TOKENS,
                        # temperature = 0.7, # default value
                        temperature = config.TEMPERATURE,
                        # top_k = 50, # default value
                        top_k = config.TOP_K,
                        # top_p = 0.7, # default value
                        top_p = config.TOP_P,
                        repetition_penalty = config.REPETITION_PENALTY,
                        stop = ['\n\n']
                        )
    json_object = json.dumps(summary_output, indent=4)
    outfile = open(f'./outputs/output_{i}.json', 'w+')
    outfile.write(json_object)
    outfile.close()
    outputs.append(summary_output)
    print(f'Story {i} summarized and saved.')
    i+=1
    time.sleep(1)

print('*'*100)
# pprint(outputs)

# save outputs
# json_object = json.dumps(outputs, indent=4)

# with open("/output/outputs.json", "w") as outfile:
#     outfile.write(json_object)

output_strings = []
for out in outputs:
    output_strings.append(out['output']['choices'][0]['text'])

print('*'*100)
# print(output_strings)
end_time = time.time()

print('Time taken:', end_time - start_time)

# print generated text
# print(output['prompt'][0]+output['output']['choices'][0]['text'])

# stop
# together.Models.stop("togethercomputer/llama-2-7b-chat")
together.Models.stop(config.MODEL)
print(f'Model {config.MODEL} stopped.')

# # check which models have started or stopped
# together.Models.instances()
