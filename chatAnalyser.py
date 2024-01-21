import re
import emoji
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from collections import Counter
from PIL import Image

nltk.download('punkt')
nltk.download('vader_lexicon')

def extract_emojis(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"
                           u"\U0001F300-\U0001F5FF"
                           u"\U0001F680-\U0001F6FF"
                           u"\U0001F700-\U0001F77F"
                           u"\U0001F780-\U0001F7FF"
                           u"\U0001F800-\U0001F8FF"
                           u"\U0001F900-\U0001F9FF"
                           u"\U0001FA00-\U0001FA6F"
                           u"\U0001FA70-\U0001FAFF"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.findall(text)

def emojis_to_images(emojis):
    images = {}
    for emoji_unicode in emojis:
        emoji_filename = f"emoji_images/{emoji_unicode}.png"
        try:
            image = Image.open(emoji_filename)
            images[emoji_unicode] = image
        except FileNotFoundError:
            print(f"Emoji image not found: {emoji_unicode}")

    return images

def count_specific_sentence(chat_text, sender_sentence, receiver_sentence):
    sender_matches = re.findall(fr'\b{re.escape(sender_sentence)}\b', chat_text, flags=re.IGNORECASE)
    receiver_matches = re.findall(fr'\b{re.escape(receiver_sentence)}\b', chat_text, flags=re.IGNORECASE)
    return len(sender_matches), len(receiver_matches)

def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)['compound']

    if sentiment_score >= 0.05:
        return 'Positive'
    elif sentiment_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def analyze_word_frequency(text, top_n=10):
    words = word_tokenize(text.lower())
    word_counts = Counter(words)
    top_words = word_counts.most_common(top_n)
    return top_words

def main():
    with open('exported_chat.txt', 'r', encoding='utf-8') as file:
        chat_text = file.read()

    sender_sentence = input("Enter the sender sentence to count: ")
    receiver_sentence = input("Enter the receiver sentence to count: ")

    emojis = Counter(extract_emojis(chat_text))
    sender_count, receiver_count = count_specific_sentence(chat_text, sender_sentence, receiver_sentence)
    sentiment = analyze_sentiment(chat_text)
    top_words = analyze_word_frequency(chat_text)

    # Print analysis results
    print("Emojis:")
    for emoji_unicode, count in emojis.items():
        print(f"{emoji_unicode}: {count}")

    print(f"Sender Sentence '{sender_sentence}': {sender_count} occurrences")
    print(f"Receiver Sentence '{receiver_sentence}': {receiver_count} occurrences")
    print(f"Sentiment: {sentiment}")
    print("Top Words:")
    for word, count in top_words:
        print(f"{word}: {count}")

if __name__ == "__main__":
    main()
