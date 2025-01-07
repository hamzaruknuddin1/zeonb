from flask import Flask, request, jsonify
from twilio.twiml.voice_response import VoiceResponse
from twilio.rest import Client
import os
import openai
from dotenv import load_dotenv
import logging
from scrape_and_embeddings import (
    scrape_website_advanced,
    create_chunks_with_token_limit,
    create_embeddings,
    store_embeddings_faiss,
    retrieve_context
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Company Configuration
COMPANY_CONFIG = {
    "name": os.getenv("COMPANY_NAME", "Default Company"),
    "website": os.getenv("COMPANY_WEBSITE", "https://example.com"),
    "receptionist_name": os.getenv("RECEPTIONIST_NAME", "Alex"),
    "voice": os.getenv("VOICE_NAME", "Polly.Kendra"),  # Twilio voice
    "speaking_speed": os.getenv("SPEAKING_SPEED", "fast"),
    "default_context": os.getenv("DEFAULT_CONTEXT", "We provide professional services to our clients.")
}

# API Configuration
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")
DEFAULT_PHONE = os.getenv("DEFAULT_PHONE")

openai.api_key = OPENAI_API_KEY

app = Flask(__name__)
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Global conversation context
conversation_contexts = {}

def initialize_embeddings():
    """Initialize embeddings and FAISS index."""
    logging.info(f"Starting website scraping for {COMPANY_CONFIG['name']}")
    try:
        scraped_content = scrape_website_advanced(COMPANY_CONFIG['website'], use_selenium=True)
        
        if not scraped_content:
            logging.error("No content scraped from website")
            return None, None
            
        chunks = create_chunks_with_token_limit(scraped_content)
        
        if not chunks:
            logging.error("No valid chunks created")
            return None, None

        logging.info(f"Created {len(chunks)} chunks")
        
        embeddings = create_embeddings(chunks)
        if embeddings is None:
            logging.error("Failed to create embeddings")
            return None, None
            
        faiss_index = store_embeddings_faiss(embeddings)
        
        logging.info("Successfully initialized embeddings and FAISS index")
        return chunks, faiss_index
        
    except Exception as e:
        logging.error(f"Error during initialization: {str(e)}")
        return None, None

# Initialize global variables
chunks, faiss_index = initialize_embeddings()
if chunks is None or faiss_index is None:
    logging.error("Failed to initialize embeddings and FAISS index")
    chunks = [COMPANY_CONFIG['default_context']]

def create_voice_response():
    """Create a TwiML VoiceResponse object."""
    return VoiceResponse()

def say_with_voice(response, text):
    """Add a <Say> verb with the configured voice and speaking speed."""
    ssml_text = f'<speak><prosody rate="{COMPANY_CONFIG["speaking_speed"]}">{text}</prosody></speak>'
    response.say(ssml_text, voice=COMPANY_CONFIG["voice"], language="en-US", ssml=True)

@app.route("/call", methods=["POST"])
def start_call():
    """Initiate the call."""
    data = request.json
    customer_phone = data.get('phone_number', DEFAULT_PHONE)

    if not customer_phone:
        return jsonify({"error": "Missing phone number."}), 400

    call = twilio_client.calls.create(
        to=customer_phone,
        from_=TWILIO_PHONE_NUMBER,
        url=f"{BASE_URL}/conversation",
        status_callback=f"{BASE_URL}/call_status",
        status_callback_event=["initiated", "ringing", "answered", "completed"],
        machine_detection="Enable",
    )

    return jsonify({"message": "Call initiated", "call_sid": call.sid}), 200

@app.route("/call_status", methods=["POST"])
def call_status():
    """Handle call status updates."""
    call_status = request.values.get("CallStatus")
    answered_by = request.values.get("AnsweredBy")

    logging.info(f"Call status: {call_status}")
    if call_status == "completed":
        logging.info(f"Call completed. Answered by: {answered_by}")

    return "", 200

@app.route("/conversation", methods=["POST", "GET"])
def conversation():
    """Manage the dynamic call flow."""
    session_id = request.args.get("session_id", "default_session")
    response = create_voice_response()

    if session_id not in conversation_contexts:
        conversation_contexts[session_id] = {
            "history": [],
            "awaiting_input": False,
        }

    context = conversation_contexts[session_id]

    if not context["awaiting_input"]:
        context["awaiting_input"] = True
        gather = response.gather(
            input="speech",
            action=f"{BASE_URL}/handle_response?session_id={session_id}",
            timeout=8,
            speech_timeout="auto",
            enhanced=True,
            speech_model="phone_call",
        )
        greeting = f"Hello, this is {COMPANY_CONFIG['receptionist_name']} from {COMPANY_CONFIG['name']}. How may I assist you today?"
        say_with_voice(gather, greeting)
        return str(response)

    return str(response)

@app.route("/handle_response", methods=["POST"])
def handle_response():
    """Handle user response from Twilio."""
    session_id = request.args.get("session_id", "default_session")
    user_response = request.form.get("SpeechResult", "").strip()

    if not user_response:
        response = create_voice_response()
        say_with_voice(response, "I'm sorry, I didn't hear that. Could you please repeat?")
        response.redirect(f"{BASE_URL}/conversation?session_id={session_id}")
        return str(response)

    logging.info(f"User Response: {user_response}")

    context = conversation_contexts[session_id]

    try:
        # Retrieve relevant context
        relevant_context = retrieve_context(user_response, faiss_index, chunks)
        
        # Build history-aware context
        history = "\n".join([f"User: {entry['user']}\nAI: {entry['ai']}" for entry in context["history"]])
        history_with_context = f"{relevant_context}\n\nConversation History:\n{history}" if history else relevant_context

        # Generate AI response
        ai_response = generate_ai_response(history_with_context, user_response)
        logging.info(f"AI Response: {ai_response}")

        # Append to history
        context["history"].append({"user": user_response, "ai": ai_response})
    except Exception as e:
        logging.error(f"Error during context retrieval or response generation: {e}")
        ai_response = "I'm sorry, I couldn't understand. Could you please repeat?"

    response = create_voice_response()

    if "[END_CALL]" in ai_response:
        say_with_voice(response, f"Thank you for calling {COMPANY_CONFIG['name']}. Goodbye!")
        response.hangup()
        return str(response)

    gather = response.gather(
        input="speech",
        action=f"{BASE_URL}/handle_response?session_id={session_id}",
        timeout=8,
        speech_timeout="auto",
        enhanced=True,
        speech_model="phone_call",
    )
    say_with_voice(gather, ai_response)
    return str(response)

def generate_ai_response(context, user_question):
    """Generate AI response using GPT."""
    if not context:
        context = COMPANY_CONFIG['default_context']
        
    prompt = f"""
    You are {COMPANY_CONFIG['receptionist_name']}, a friendly and professional receptionist at {COMPANY_CONFIG['name']}.
    Your job is to help customers with any questions about {COMPANY_CONFIG['name']}, including services, pricing, and packages.

    Context about the company:
    {context}

    Current User Question: {user_question}

    Instructions:
    - Respond concisely and professionally as {COMPANY_CONFIG['receptionist_name']} in 1-2 sentences for fast processing.
    - If you do not understand the user's question, ask them politely to repeat or clarify.
    - Use `[END_CALL]` only if the user explicitly wants to end the call or expresses no further interest.
    - Do not end the call unless the user explicitly indicates they are done.

    Answer as {COMPANY_CONFIG['receptionist_name']}, keeping responses polite and professional:
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.7,
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        logging.error(f"Error during GPT generation: {e}")
        return "I apologize, but I'm having trouble processing your question. Could you please repeat that?"

if __name__ == "__main__":
    app.run(port=5000)