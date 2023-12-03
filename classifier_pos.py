import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize

# Download NLTK data for part-of-speech tagging
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Read the CSV file into a pandas dataframe
file_path = './data/combined_70b.csv'
df = pd.read_csv(file_path)

# Function to preprocess the 'summary' column and handle NaN values
def preprocess_summary(text):
    if pd.isnull(text):
        return ""
    return text

# Apply the preprocessing function to the 'summary' column
df['summary'] = df['summary'].apply(preprocess_summary)

# Function to calculate part-of-speech distribution of each summary
def calculate_pos_distribution(text):
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    # print('pos_tags')
    # print(pos_tags)
    # exit()
    
    pos_counts = {}
    for _, pos in pos_tags:
        pos_counts[pos] = pos_counts.get(pos, 0) + 1
    
    total_tokens = len(tokens)
    pos_distribution = {pos: count / total_tokens for pos, count in pos_counts.items()}
    
    return pos_distribution

# Add new columns for each part of speech to the dataframe
pos_columns = ['NN', 'CC', 'PRP', 'VBD', 'TO', 'DT', 'RB', 'PRP$', '.', 'VBG', 'IN', 'NNS', 'JJ']
for pos in pos_columns:
    df[f'{pos.lower()}_ratio'] = df['summary'].apply(lambda x: calculate_pos_distribution(x).get(pos, 0))

# Split the data into training (80%) and testing (20%) sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

train_df.to_csv('./data/train_stratify_pos.csv', index=False)
test_df.to_csv('./data/test_stratify_pos.csv', index=False)

# Select the part-of-speech ratio columns as features for training
X_train = train_df[[f'{pos.lower()}_ratio' for pos in pos_columns]].values
y_train = train_df['label']

# Train a logistic regression classifier using part-of-speech distribution
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Select the same columns for the testing set
X_test = test_df[[f'{pos.lower()}_ratio' for pos in pos_columns]].values
y_test = test_df['label']

# Test the classifier on the validation set
predictions = classifier.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')
