# University Academic Advisor Chatbot 🎓

This project is a Multilingual RAG (Retrieval-Augmented Generation) Chatbot designed to help students and staff navigate university regulations and standards. It supports English, French, Arabic, and Moroccan Darija.

## Features

- **PDF Processing**: Upload academic documents (e.g., student handbooks, regulations) and index them for querying.
- **Multilingual Support**: Supports English, French, Arabic, and Moroccan Darija. The chatbot detects the input language and responds accordingly.
- **RAG Architecture**: Uses LangChain and FAISS for efficient document retrieval and Google Gemini for high-quality response generation.
- **Evaluation**: Includes evaluation scripts using RAGAS to measure faithfulness, answer relevance, and context precision.

## Tech Stack

- **Frontend**: Streamlit
- **LLM**: Google Gemini 1.5 Flash (via `langchain-google-genai`)
- **Embeddings**: HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
- **Vector Store**: FAISS
- **RAG Framework**: LangChain
- **Regex Cleaning**: Custom utility for cleaning academic text.

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd NLP-Project
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate  # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory based on `.env.example`:
```bash
cp .env.example .env
```
Fill in your API keys:
- `GOOGLE_API_KEY`: Your Google AI Studio API Key.
- `GROQ_API_KEY`: (Optional) Your Groq API Key if using Groq models.

### 5. Run the Application
```bash
streamlit run src/ui.py
```

## Project Structure

- `src/`: Core application logic and UI.
  - `engine.py`: RAG pipeline and PDF processing.
  - `ui.py`: Streamlit interface.
  - `utils.py`: Text cleaning utilities.
- `assets/`: Project images and demo video.
- `data/`: Directory for uploaded PDFs (ignored by git).
- `evals/`: RAGAS evaluation scripts and results.
- `notebooks/`: Exploration and development notebooks.

## Evaluation

Evaluation results and graphs can be found in the `evals/` and `assets/` directories. Use `evals/ragas_eval.py` to run evaluations.

## Deployement

The app was deployed on Streamlit, you can check it on : https://academic-chatbot-gyseabzdwihugwyafe6igb.streamlit.app/

## License

[MIT License](LICENSE)
