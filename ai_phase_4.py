import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import re
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Download the missing NLTK resource
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
dataset_path = 'archive/dialogs.txt'  # Replace with the actual file path
data = pd.read_csv(dataset_path, delimiter='\t', names=['User', 'Bot'])

# Display the first few rows of the dataset
print(data.head())

# Check for missing values
missing_values = data.isnull().sum()
print("Missing Values:")
print(missing_values)

# Check the distribution of user and bot messages
user_message_count = len(data[data['User'].notnull()])
bot_message_count = len(data[data['Bot'].notnull()])
total_messages = len(data)
user_message_percentage = (user_message_count / total_messages) * 100
bot_message_percentage = (bot_message_count / total_messages) * 100
print(f"Total Messages: {total_messages}")
print(f"User Messages: {user_message_count} ({user_message_percentage:.2f}%)")
print(f"Bot Messages: {bot_message_count} ({bot_message_percentage:.2f}%)")

# Preprocess the training data
def preprocess_text(text):
    # Tokenization
    words = word_tokenize(text)
    # Removing stopwords
    words = [word for word in words if word not in stopwords.words('english')]
    # Join the words back into a text
    return ' '.join(words)

# Apply preprocessing to your documents
X = data['User'].apply(preprocess_text)  # Preprocess the 'User' column

# Vectorize text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(min_df=1)  # Adjust min_df as needed
X = tfidf_vectorizer.fit_transform(X)

# Convert text labels to binary labels (0 or 1)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['Bot'])

# Train the model for binary classification
model = MultinomialNB()
model.fit(X, y)

# Define the chat_with_bot function
def chat_with_bot(user_input):
    # Preprocess the user input
    user_input = user_input.lower()
    user_input = tfidf_vectorizer.transform([user_input])

    # Make a prediction using the trained model
    prediction = model.predict(user_input)

    # Convert the prediction back to a text label ('Bot' or 'Not Bot')
    predicted_label = label_encoder.inverse_transform(prediction)

    # Return the chatbot's response based on the predicted label
    if predicted_label[0] == 'Bot':
        return "Chatbot Response: [Your response for Bot here]"
    else:
        return "Chatbot Response: [Your response for Not Bot here]"
    
    model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def generate_response(user_input):
    input_ids = tokenizer.encode(user_input, return_tensors="pt")
    response_ids = model.generate(input_ids, max_length=50, num_return_sequences=1)
    response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    return response

while True:
    user_input = input("You: ")  # Get user input
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break  # Exit the loop if the user types "exit"
    
    chatbot_response = generate_response(user_input)  # Get chatbot response
    print(f"Chatbot: {chatbot_response}")  # Display chatbot's response

