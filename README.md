# LangChain Documentation Helper Chatbot - Helper Bot
## A Python-based documentation helper that answers marketing/sales questions 

## Features
- **File Ingestion to VectorStore**: Ingestion of files into a any vectorstore (Pinecone for now) for search and retrieval capabilities within your documentation.
- **Natural Language Processing**: LangChain facilitates the creation of complex workflows by chaining together different LLM related operations
- **Customizable Templates**: Usage of custom templates to pass as prompts to LLM.

## Installation

- Clone the repository 
- Manually add .env file to the project (running the project without the API keys is impossible)
### Three ways to work with the project (to be updated constantly...)

### Using Global Python Installation
- Ensure Python 3.11 or later is installed on your system.
- Make sure you have pip installed on your computer
- To install the needed packages: pip install -r requirements.txt
- To run the main file of the program: streamlit run main.py

### Using a Virtual Environment (pyenv)
- Make sure to correctly install pyenv according to this article (Ubuntu >= 22.04 preffered):
https://medium.com/@aashari/easy-to-follow-guide-of-how-to-install-pyenv-on-ubuntu-a3730af8d7f0
- After installing pyenv, install any desired version of python: pyenv install 3.11.1
- Make sure you have pip installed on your computer
- Create the virtual environmen: pyenv virtualenv 3.12.1 myenv
- Activate environment: pyenv activate myenv
- To install the needed packages: pip install -r requirements.txt
- To run the main file of the program: streamlit run main.py
- To deactivate environment: pyenv deactivate

### Using a Docker Image
- There is a Dockerfile inside the repo
- To build a local image: docker build -t image_name .
- To run the image: docker run -p 8501:8501 image_name
