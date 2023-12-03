import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
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

# Split the data into training (80%) and testing (20%) sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
train_df.to_csv('./data/train_stratify_tfidf.csv', index=False)
test_df.to_csv('./data/test_stratify_tfidf.csv', index=False)

# Create a TfidfVectorizer to convert text to a matrix of TF-IDF features
vectorizer = TfidfVectorizer()

# Fit and transform the training data
X_train = vectorizer.fit_transform(train_df['summary'])

# Train a logistic regression classifier
classifier = LogisticRegression()
classifier.fit(X_train, train_df['label'])

# Transform the test data
X_test = vectorizer.transform(test_df['summary'])

# Make predictions on the test data
predictions = classifier.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(test_df['label'], predictions)
print(f'Accuracy: {accuracy}')
