
# Medical Chatbot

This project is a medical chatbot created with LangChain and Hugging Face, utilizing the `llama-2-7b-chat.ggmlv3.q2_K` model. The chatbot performs similarity searches on medical documents using embeddings and provides relevant responses.

## Project Structure

- **Environment**: Python 3.10
- **Dependencies**: Installed via `requirements.txt`
- **Files & Directories**:
  - `setup.py` - Sets up the package for deployment
  - `templates.py` - Creates the file structure
  - `document_loaders` - Loads and processes medical documents
  - `app.py` - Flask application file for routing

## Key Steps

1. **Environment Setup**
   - Created a new Python 3.10 environment.
   - Installed dependencies using `requirements.txt`.

2. **File Structure**
   - Created using `setup.py` and `templates.py`.

3. **Document Loading**
   - Loaded medical documents using LangChain's `directory_loader` and `pyPDFLoader`.
   - Extracted text from PDFs and segmented it into manageable chunks.

4. **Embeddings and Vector Database**
   - Used Hugging Face's `"sentence-transformers/all-MiniLM-L6-v2"` model (384-dimensional) for embeddings.
   - Initialized Pinecone to create a vector database and store the embeddings.
   - Uploaded vectors to the Pinecone vector space, enabling similarity searches.

5. **Prompt Creation and Language Model (LLM) Generation**
   - Created a prompt and generated the LLM with the `ctransformers` library using the local model.
   - Verified outputs for accuracy in responses.

6. **Flask Frontend**
   - Developed a frontend using Flask and defined routes for chatbot interaction.
   - Integrated all components to complete the application.

## Usage

1. **Start the Application**:
   ```bash
   python app.py
   ```

2. **Interact with the Chatbot**:
   - Access the chatbot through the specified route.
   - Enter medical queries to receive relevant document-based responses.

## Future Enhancements

- Add advanced natural language understanding features.
- Integrate additional medical data sources.

## License

This project is licensed under the MIT License.
