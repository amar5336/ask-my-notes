\# Ask My Notes



A PDF question-answering app using Retrieval-Augmented Generation (RAG). Upload a PDF, ask a question, and get an answer based only on the document.



Live demo: https://ask-my-notes-jhti42hgrxt6uvnfyxpymr.streamlit.app



\## How it works

1\. Extract text from the PDF with pypdf

2\. Split it into overlapping 500-character chunks

3\. Convert chunks to embeddings with sentence-transformers

4\. Store them in a FAISS index and retrieve the top 3 matches for a question

5\. Send the matches and the question to an LLM through the Groq API



\## Tech stack

Python, Streamlit, pypdf, sentence-transformers, FAISS, Groq API



\## Results

Answered 8 out of 10 test questions correctly (80% accuracy) on a 10-question test set.



\## Run locally

pip install -r requirements.txt

streamlit run app.py



\## Author

Amar | GitHub: amar5336

