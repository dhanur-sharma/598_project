import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from transformers import pipeline

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

# Function to calculate perplexity of each summary
def calculate_perplexity(text):
    lm = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B", device=0)
    perplexity = lm(text, max_length=len(text), return_dict=True)['perplexity']
    return perplexity

# Add a new column 'perplexity' to the dataframe
df['perplexity'] = df['summary'].apply(calculate_perplexity)

# Split the data into training (80%) and testing (20%) sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

train_df.to_csv('./data/train_stratify_perplexity.csv', index=False)
test_df.to_csv('./data/test_stratify_perplexity.csv', index=False)
               
# Select the 'perplexity' column as a feature for training
X_train = train_df['perplexity'].values.reshape(-1, 1)
y_train = train_df['label']

# Train a logistic regression classifier using perplexity
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Select the same column for the testing set
X_test = test_df['perplexity'].values.reshape(-1, 1)
y_test = test_df['label']

# Test the classifier on the validation set
predictions = classifier.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')
