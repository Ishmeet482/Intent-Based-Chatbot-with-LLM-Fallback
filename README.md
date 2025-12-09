# Intent-Based Chatbot with LLM Fallback

A modular chatbot system combining local intent classification with a HuggingFace LLM fallback for low-confidence predictions. The chatbot processes user input through an intent engine, confidence scoring, fallback routing, and response generation. Includes configurable thresholds, structured logging, and clean separation between logic (chatbot.py) and execution (new.py).

## Features
### Intent Classification

-**Custom Intent Engine**: Classifies user messages using examples defined in intents.json.

-**Confidence Scoring**: Determines when to use a predefined response vs. fallback to LLM.

-**Expandable Intent Dataset**: Add new intents easily by editing intents.json.

-**Fast Local Matching**: Efficient keyword/similarity-based scoring.

### LLM Fallback (HuggingFace)

-**Low-Confidence Fallback**: When intent score is below threshold, chatbot uses a HuggingFace model.

-**Context-Aware Replies**: Uses generative models for natural, longer responses.

-**API-Based or Local Model**: Works with hosted inference endpoints or local transformers.

-**Configurable API Key**: Loaded through environment variables or .env.

### Core Chatbot Engine

-**Modular Architecture**: Logic in chatbot.py, execution in new.py.

-**Structured Logging**: Tracks user messages, detected intents, and model decisions.

-**Separation of Concerns**: Clean workflow for classification → routing → responding.


## Architecture
```
┌──────────────────────────────────────────────────────────────┐
│                         Chatbot Engine                       │
│  (Intent detection, confidence scoring, fallback routing)    │
└─────────────────────┬────────────────────────────────────────┘
                      │
     ┌────────────────┼───────────────────────────────┐
     │                │                               │
     ▼                ▼                               ▼
┌───────────┐  ┌──────────────┐             ┌──────────────────┐
│ Local      │  │ Confidence   │             │ HuggingFace LLM  │
│ Intent     │  │ Thresholding │             │  (Fallback)      │
│ Classifier │  └──────────────┘             └──────────────────┘
└───────────┘          │                               │
                       ▼                               ▼
                ┌──────────────────────────────────────────┐
                │               Responder                  │
                │ (Template replies / Model responses)     │
                └──────────────────────────────────────────┘
```
## Getting Started
### Prerequisites

-Python 3.8+

-HuggingFace API key (if using hosted inference endpoint)

-Virtual environment recommended

### Installation
# Clone repository
```bash
git clone https://github.com/<your-username>/Intent-Based-Chatbot-with-LLM-Fallback.git
cd Intent-Based-Chatbot-with-LLM-Fallback
```
# Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\Activate.ps1 # Windows
```
# Install dependencies
```bash
pip install -r requirements.txt
```
Insert Your API Key
Option 1 — Using Environment Variable (Recommended)
export HUGGINGFACE_API_KEY="hf_xxxYOURKEYxxx"

Option 2 — Using .env File

Create .env:

HUGGINGFACE_API_KEY=hf_xxxYOURKEYxxx


Ensure chatbot.py loads it:
```basg
from dotenv import load_dotenv
load_dotenv()
```
Option 3 — Hardcode (Not Recommended)

Directly inside chatbot.py:
```bash
HF_API_KEY = "hf_xxxYOURKEYxxx"
```
Running the Chatbot
Start via new.py (recommended)
```bash
python new.py
```
Or run core logic directly:
python chatbot.py


## Expected output includes logs such as:
```basg
INFO: User message: hi
INFO: Top local intent: greeting (conf=0.78)
Great! Bot is Running!
```
## Core Workflows
### 1. Intent Detection
```bash
intent, confidence = classifier.predict("hello")
```
### 2. Fallback Routing
```bash
if confidence < threshold:
    response = llm.generate(user_input)
else:
    response = predefined_response[intent]
```
### 3. Chat Loop (CLI)
```bash
while True:
    user = input("You: ")
    print("Bot:", chatbot.respond(user))

Extending Intents

Add a new intent:

{
  "tag": "motivation",
  "patterns": ["motivate me", "give me motivation"],
  "responses": ["You can do it! Stay strong."]
}


Update intents.json and the classifier handles it automatically.
```
## Project Structure
```
Intent-Based-Chatbot-with-LLM-Fallback/
├── chatbot.py               # Core chatbot engine
├── new.py                   # Runner / model loader
├── intents.json             # Intent dataset
├── requirements.txt         # Dependencies
├── README.md                # Documentation
└── .env.example             # Example environment variable file (optional)
```

## Testing

You can run basic tests manually:
```bash
python chatbot.py    # For engine validation
python new.py        # For full runtime validation
```

## Demo Interaction
You: hi
Bot: Hello! How can I help?

You: explain quantum entanglement
Bot: (LLM-generated explanation based on fallback)
