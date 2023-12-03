FILE_PATH = "/common/home/ds1987/598_project/598_project/data/hcV3-stories.csv"

# INSTRUCTION = "Summarize the following story in 3 sentences. Don't output anything but the summary generated. Write it in first person. : "
# INSTRUCTION = "Summarize the following story in first person: "
INSTRUCTION = "Summarize the following story in first person in 3 sentences: "


# MODEL = 'togethercomputer/llama-2-7b-chat'
# MODEL = 'togethercomputer/llama-2-13b-chat'
# MODEL = 'Austism/chronos-hermes-13b'
# MODEL = 'togethercomputer/GPT-NeoXT-Chat-Base-20B'
# MODEL = 'togethercomputer/llama-2-70b'
MODEL = 'togethercomputer/llama-2-70b-chat'

# MODEL_PARAMETERS
MAX_TOKENS = 256
TEMPERATURE = 0.8
TOP_K = 60
TOP_P = 0.8
REPETITION_PENALTY = 1


STORY_START = 0
# STORY_START = 3
# STORY_START = 10
# STORY_START = 1583


# NUMBER_OF_STORIES = 3
# NUMBER_OF_STORIES = 10
NUMBER_OF_STORIES = 6854 # default value - contains all stories



"""7b removed stop

STORY_START = 0
# STORY_START = 3
# STORY_START = 10
# STORY_START = 1583


# NUMBER_OF_STORIES = 3
# NUMBER_OF_STORIES = 10
NUMBER_OF_STORIES = 6854 # default value - contains all stories
"""

"""
chronos-hermes-13b
# STORY_START = 0
# STORY_START = 2691
STORY_START = 5626

NUMBER_OF_STORIES = 6854 # default value - contains all stories

"""

"""
13b
# STORY_START = 0
# STORY_START = 9
STORY_START = 5226

NUMBER_OF_STORIES = 6854 # default value - contains all stories
"""

"""
# 7b
STORY_START = 0
# STORY_START = 1836
# STORY_START = 2741
# STORY_START = 2890
# NUMBER_OF_STORIES = 20
NUMBER_OF_STORIES = 6854 # default value - contains all stories
"""
