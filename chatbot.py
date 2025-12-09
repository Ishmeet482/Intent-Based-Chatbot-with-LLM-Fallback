import os
import random
import json
import pickle
import logging
import numpy as np
import nltk
from typing import Optional

import logging
logging.disable(logging.CRITICAL)


from nltk.stem import WordNetLemmatizer
from keras.models import load_model

# Optional: configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Hugging Face/OpenAI router client setup ---
# Requires: pip install openai
try:
    from openai import OpenAI
except Exception as e:
    OpenAI = None
    logger.warning("OpenAI package not available. Hugging Face fallback won't work until you `pip install openai`. Error: %s", e)

def create_hf_client():
    """
    Create and return an OpenAI-compatible client that routes to Hugging Face.
    Returns None if token missing or package not installed.
    """
    if OpenAI is None:
        return None
    hf_token = "Insert Your API Key Here"
    if not hf_token:
        logger.info("HF_TOKEN not set in environment; HF fallback disabled.")
        return None
    try:
        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=hf_token,
        )
        return client
    except Exception as e:
        logger.exception("Failed to create HF/OpenAI client: %s", e)
        return None

_HF_CLIENT = create_hf_client()
HF_MODEL_DEFAULT = os.environ.get("HF_MODEL", "deepseek-ai/DeepSeek-V3.2:novita")

def extract_text_from_completion(completion) -> Optional[str]:
    """
    Safely extract a text reply from various completion shapes returned by the client.
    Returns None if extraction fails.
    """
    try:
        # Try object-like access (OpenAI SDK)
        if hasattr(completion, "choices") and len(completion.choices) > 0:
            choice = completion.choices[0]
            # choice.message.content
            msg = getattr(choice, "message", None)
            if msg:
                # msg might be dict-like or object-like
                content = getattr(msg, "content", None)
                if content:
                    return content
                if isinstance(msg, dict):
                    return msg.get("content") or msg.get("text")
            # other fallback: choice.text
            txt = getattr(choice, "text", None)
            if txt:
                return txt
            # dict-like choice
            if isinstance(choice, dict):
                return choice.get("message", {}).get("content") or choice.get("text")
        # dict-like top-level
        if isinstance(completion, dict):
            # completion may be {'choices': [{'message': {'content': '...'}}]}
            choices = completion.get("choices", [])
            if choices:
                c0 = choices[0]
                if isinstance(c0, dict):
                    return c0.get("message", {}).get("content") or c0.get("text")
            # try top-level 'text'
            if "text" in completion:
                return completion["text"]
        # final fallback
        return str(completion)
    except Exception as e:
        logger.exception("Error extracting text from completion: %s", e)
        return None

def get_hf_response(user_text: str, model: str = HF_MODEL_DEFAULT) -> Optional[str]:
    """
    Send user_text to Hugging Face router via OpenAI-compatible SDK and return a text reply.
    Returns None on failure or if HF client not configured.
    """
    global _HF_CLIENT
    if _HF_CLIENT is None:
        _HF_CLIENT = create_hf_client()
    if _HF_CLIENT is None:
        return None

    try:
        completion = _HF_CLIENT.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user_text}],
        )
        return extract_text_from_completion(completion)
    except Exception as e:
        logger.exception("Error calling HF model: %s", e)
        return None

# --- End HF setup ---

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('/Users/kishan/Documents/CHATBOT/intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]), verbose=0)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    """
    Return a response from intents.json for the top intent.
    If intents_list empty -> returns None.
    """
    if not intents_list:
        return None
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            return result
    return None

#print("GO! Bot is running!")

import re
QUESTION_WORDS = {"what", "when", "where", "who", "why", "how", "which", "whom", "whose"}

def looks_like_question(text: str) -> bool:
    # Simple heuristic: starts with question word or ends with '?'
    if text.strip().endswith('?'):
        return True
    first = text.strip().split(' ')[0].lower() if text.strip() else ''
    if first in QUESTION_WORDS:
        return True
    # contains "?" anywhere or "tell me about" etc.
    if '?' in text or re.search(r'\btell me about\b', text.lower()):
        return True
    return False

# Tweak threshold here
LOCAL_CONF_THRESHOLD = 0.60

print("GO! Bot is running (with HF fallback).")

while True:
    try:
        message = input("").strip()
        if not message:
            continue

        # Local prediction
        ints = predict_class(message)
        top_conf = 0.0
        top_intent = None
        if ints:
            try:
                top_conf = float(ints[0].get('probability', "0"))
                top_intent = ints[0].get('intent')
            except Exception:
                top_conf = 0.0

        # Debug info to help you diagnose what's happening
        logger.info("User message: %s", message)
        logger.info("Top local intent: %s (conf=%s)", top_intent, top_conf)
        logger.info("HF client present? %s", bool(_HF_CLIENT))

        # Decision logic:
        # 1) If local confidence is high, use local response
        # 2) If local confidence is low OR message looks like a question OR local intent is a generic tag, use HF fallback
        GENERIC_INTENTS = {"query", "unknown", "fallback", "thanks"}  # modify as needed
        use_local = False
        if ints and top_conf >= LOCAL_CONF_THRESHOLD and top_intent not in GENERIC_INTENTS:
            use_local = True

        # If message looks strongly like a question, prefer HF when local is weak
        if not use_local and looks_like_question(message):
            logger.info("Message looks like a question and local not confident -> will try HF first")

        # Try HF if local not confident or generic intent
        if not use_local:
            hf_reply = get_hf_response(message)
            if hf_reply:
                # Print HF reply if available
                print(hf_reply)
                continue
            else:
                logger.warning("HF reply unavailable or failed; falling back to local")

        # If we reach here, attempt to use local response (even if low confidence)
        if ints:
            res = get_response(ints, intents)
            if res:
                print(res)
                continue

        # Final fallback
        print("Sorry, I didn't understand that. Could you rephrase?")
    except KeyboardInterrupt:
        print("\nExiting. Bye!")
        break
    except Exception as e:
        logger.exception("Unexpected error in main loop: %s", e)
        print("Oops, something went wrong. Try again.")
