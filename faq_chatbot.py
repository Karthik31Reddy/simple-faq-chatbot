import nltk
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Sample FAQ data
faq_data = {
    "What is your return policy?": "You can return any item within 30 days of purchase.",
    "How long does shipping take?": "Shipping typically takes 5-7 business days.",
    "Do you offer international shipping?": "Yes, we ship to most countries worldwide.",
    "How can I track my order?": "You can track your order through the tracking link in your confirmation email.",
    "What payment methods are accepted?": "We accept Visa, Mastercard, PayPal, and Apple Pay.",
    "Can I pick up my order in-store?": "Yes, we offer in-store pickup at select locations.",
    "What should I do if I receive a damaged item?": "Please contact customer service within 48 hours of receiving the item.",
    "How do I reset my password?": "Click on 'Forgot Password' at login and follow the instructions.",
    "Can I change my order after placing it?": "Yes, within 1 hour of placing the order.",
    "How do I contact customer service?": "You can reach us via email at xyz@gmail.com.",
    "What are your business hours?": "We are open Monday to Friday, 9 AM to 5 PM EST.",
    "Do you have a loyalty program?": "Yes, we offer a loyalty program that rewards you with points for every purchase.",
    "Where is your company located?": "We are based in New York, USA.",
    "Is my personal information secure?": "Yes, we use industry-standard encryption to protect your data."
}

# Preprocessing function
def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in string.punctuation and t not in stop_words]
    return " ".join(tokens)

# Prepare data
questions = list(faq_data.keys())
processed_questions = [preprocess(q) for q in questions]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processed_questions)

# Chat function
def get_response(user_input):
    processed_input = preprocess(user_input)
    user_vec = vectorizer.transform([processed_input])
    similarities = cosine_similarity(user_vec, tfidf_matrix)
    best_match_index = similarities.argmax()
    if similarities[0, best_match_index] < 0.2:
        return "Sorry, I couldn't understand your question. Could you please rephrase?"
    return faq_data[questions[best_match_index]]

# Chat loop
print("Hi! I'm your FAQ bot. Ask me a question or type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Bot: Goodbye!")
        break
    response = get_response(user_input)
    print("Bot:", response)
