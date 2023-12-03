import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
from nltk.tokenize import sent_tokenize

# Download the nltk punkt package for sentence tokenization
nltk.download('punkt')

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

# Function to calculate average sentence length of each summary
def calculate_avg_sentence_length(text):
    sentences = sent_tokenize(text)
    if len(sentences) == 0:
        return 0
    total_sentence_length = sum(len(sentence.split()) for sentence in sentences)
    return total_sentence_length / len(sentences)

# Add a new column 'avg_sentence_length' to the dataframe
df['avg_sentence_length'] = df['summary'].apply(calculate_avg_sentence_length)

# Split the data into training (80%) and testing (20%) sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

train_df.to_csv('./data/train_stratify_avg_sentence_length.csv', index=False)
test_df.to_csv('./data/test_stratify_avg_sentence_length.csv', index=False)

# Train a logistic regression classifier using average sentence length
X_train = train_df['avg_sentence_length'].values.reshape(-1, 1)
y_train = train_df['label']

classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Test the classifier on the validation set
X_test = test_df['avg_sentence_length'].values.reshape(-1, 1)
y_test = test_df['label']
predictions = classifier.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')
