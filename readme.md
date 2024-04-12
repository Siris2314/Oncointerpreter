# Oncointerpreter.ai

## Enables interactive, personalized summarization of cancer diagnostics data

Oncointerpreter.ai is a cutting-edge tool designed to enhance the way healthcare professionals and researchers interpret complex cancer diagnostics data. By leveraging advanced machine learning and natural language processing technologies, Oncointerpreter.ai offers an interactive and personalized experience to its users, enabling them to gain meaningful insights from their data swiftly and accurately.

### Libraries Used

The development of Oncointerpreter.ai is supported by the integration of several powerful libraries:

- **HuggingFace**: Utilized for its robust transformer models and NLP tools, facilitating sophisticated text analysis and interpretation.
- **LangChain**: Employs LangChain for chaining together language models and other components to build complex language applications.
- **Playwright**: Utilized for end-to-end testing, ensuring the reliability and robustness of the web interface.
- **Scrapy**: Powers the data extraction capabilities, enabling the collection of relevant information from various sources.
- **Flask**: Serves as the backbone of the web application, providing a lightweight and efficient framework for the user interface.

### Description of the Files

- **llama2.py**: Contains the code for the Llama2 portion, integrating Llama2 models to enhance the analytical capabilities of the application.
- **mistral-7B.py**: Houses the Mistral-7B model implementation, contributing to the application's ability to understand and interpret complex medical texts.
- **gpt-neo.py**: Incorporates the GPT-Neo model, adding to the app's depth in generating human-like text and responses.
- **page.py**: The main UI of the app. This script initializes and runs the web interface, serving as the primary point of interaction for users.

### How to Run

To get Oncointerpreter.ai up and running on your local machine, follow these simple steps:

```bash
python3 page.py
