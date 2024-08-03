# Chatbot-RAG-LangChain-Pinecone

## Key Ideas and Techniques

- **Natural Language Processing**: Utilizes HuggingFace's language models to generate human-like responses.
- **Knowledge Base Search**: Implements a search mechanism using Pinecone to retrieve relevant context from a horoscope text file.
- **Web Interface**: Uses Streamlit to create an interactive web interface for users to chat with the bot.

## Installation

To set up the project locally, follow these steps:

1. **Clone the repository**:
   ```
   git clone https://github.com/your-username/random-fortune-telling-bot.git
   cd random-fortune-telling-bot
   ```

2. **Create and activate a virtual environment:**
   ```
   python3 -m venv venv
   source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

4. **Usage:** Please run it on **Linux**
   ```
   export PINECONE_API_KEY=your_pinecone_api_key
   export HUGGINGFACE_API_KEY=your_huggingface_api_key
   ```
   then:
   ```
   streamlit run app.py
   ```
