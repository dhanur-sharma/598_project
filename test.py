from functools import cache
from gpt4all import GPT4All
import pandas as pd
import time

import config

start_time = time.time()
def read_file(file_path):
    return pd.read_csv(file_path)

@cache
def call_model(input):
    model = GPT4All("/common/home/ds1987/598_project/gpt4all/gpt4all/wizardlm-13b-v1.1-superhot-8k.ggmlv3.q4_0.bin")
    return model.generate(input, max_tokens=20000)
    

# input = "I want you to summarize the following stories. You have to format the output separated by line separators only, and no other text in the response. Give me the summaries of the stories that you have "

df = read_file(config.FILE_PATH)

stories = df['story']
stories = stories[:3]
print(stories)

outputs = []

instruction = config.INSTRUCTION
for story in stories:
    print(story)
    prompt = instruction + str(story)
    outputs.append(call_model(prompt))
    
# print(stories)

# all_stories = '\n'.join(stories.tolist())

# print(all_stories)

# input = "I want you to summarize the following stories for me. They are separated by return characters, each one is a different story. I want you to format the output as a long string, each summary separated by a return character, and no other characters in between. Here are the stories:\n"
# input = input + all_stories
# output = call_model(input)

# print(output)
print('*'*100)
print(outputs)


end_time = time.time()

print('Time taken:', end_time - start_time)
