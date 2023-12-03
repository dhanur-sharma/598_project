import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.downloader.download('vader_lexicon')

# Read the CSV file into a pandas dataframe
file_path = './data/combined_13b.csv'
df = pd.read_csv(file_path)

# Function to preprocess the 'summary' column and handle NaN values
def preprocess_summary(text):
    if pd.isnull(text):
        return ""
    return text

# Apply the preprocessing function to the 'summary' column
df['summary'] = df['summary'].apply(preprocess_summary)

# Initialize the VADER SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

# Function to calculate sentiment intensity using VADER
def calculate_vader_sentiment(text):
    sentiment_scores = sid.polarity_scores(text)
    return sentiment_scores['compound']

# Add a new column 'sentiment_intensity' to the dataframe
df['sentiment_intensity'] = df['summary'].apply(calculate_vader_sentiment)


# Split the data into training (80%) and testing (20%) sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

train_df.to_csv('./data/train_stratify_sentiment_vader_13b.csv', index=False)
test_df.to_csv('./data/test_stratify_sentiment_vader_13b.csv', index=False)

# Train a logistic regression classifier using sentiment intensity
X_train = train_df['sentiment_intensity'].values.reshape(-1, 1)
y_train = train_df['label']

classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Test the classifier on the validation set
X_test = test_df['sentiment_intensity'].values.reshape(-1, 1)
y_test = test_df['label']
predictions = classifier.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')
