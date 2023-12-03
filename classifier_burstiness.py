import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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

# Function to calculate burstiness using coefficient of variation (CV)
def calculate_burstiness(text):
    words = text.split()
    word_frequencies = {word: words.count(word) for word in set(words)}
    
    if len(word_frequencies) == 0:
        return 0
    
    mean_frequency = sum(word_frequencies.values()) / len(word_frequencies)
    std_dev = (sum((freq - mean_frequency) ** 2 for freq in word_frequencies.values()) / len(word_frequencies)) ** 0.5
    
    # Coefficient of variation (CV)
    cv = std_dev / mean_frequency
    
    return cv

# Add a new column 'burstiness' to the dataframe
df['burstiness'] = df['summary'].apply(calculate_burstiness)

# Split the data into training (80%) and testing (20%) sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
train_df.to_csv('./data/train_stratify_burstiness.csv', index=False)
test_df.to_csv('./data/test_stratify_burstiness.csv', index=False)

# Select the 'burstiness' column as a feature for training
X_train = train_df['burstiness'].values.reshape(-1, 1)
y_train = train_df['label']

# Train a logistic regression classifier using burstiness
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Select the same column for the testing set
X_test = test_df['burstiness'].values.reshape(-1, 1)
y_test = test_df['label']

# Test the classifier on the validation set
predictions = classifier.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')
