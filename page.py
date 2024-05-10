from mistral import load_tokenizer_and_llm, load_data, process_llm_response, process_query
from flask import Flask, request, render_template, redirect, url_for, jsonify, g, session
from werkzeug.utils import secure_filename
import os
import textwrap
import time
from extract_image import extract_image
from llama2 import load_data_llama2, load_tokenizer_and_llm_llama2, process_query_llama2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.secret_key = os.urandom(24)


from pdf_extract import convert_pdf_to_text


from datetime import datetime


os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


import textwrap

def format_text_into_paragraphs(text, max_line_length=100, sentences_per_paragraph=3):
    # Split the text into sentences
    sentences = text.split('. ')

    # Group sentences into paragraphs
    paragraphs = []
    for i in range(0, len(sentences), sentences_per_paragraph):
        paragraph = '. '.join(sentences[i:i+sentences_per_paragraph])
        paragraphs.append(paragraph)

    # Wrap text to a maximum line length
    wrapped_paragraphs = [textwrap.fill(p, max_line_length) for p in paragraphs]
    
    return wrapped_paragraphs

def clean_extracted_text(text):
    # Split the text into lines
    lines = text.split('\n')

    # Remove leading and trailing spaces from each line
    cleaned_lines = [line.strip() for line in lines]

    # Remove empty lines
    cleaned_lines = [line for line in cleaned_lines if line]

    # Join the lines back into a single string
    cleaned_text = '\n'.join(cleaned_lines)

    return cleaned_text



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        start_time = datetime.now()
        selected_model = request.form.get('model')


        # Load the corresponding model and tokenizer
        if selected_model == 'llama':
            print("Llama-2 Loaded")
            llm = load_tokenizer_and_llm_llama2() 
            db = load_data_llama2()
        elif selected_model == 'mistral':
            print("Mistral Loaded")
            llm = load_tokenizer_and_llm() 
            db = load_data()
        query = request.form['query']
        # Process the query (and file if uploaded)
        response = process_query(query, llm, db)
        formatted_paragraphs = format_text_into_paragraphs(response['answer'])
        end_time = datetime.now()
        total_seconds = (end_time - start_time).total_seconds()
        file_name = session.pop('file_name', None)
        # Convert total_seconds into minutes and seconds
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        time_formatted = f"{minutes} minutes, {seconds} seconds"

        return render_template('result.html', paragraphs=formatted_paragraphs, sources=response['sources'],filename=file_name,time_taken=time_formatted,query=query)

    return render_template('index.html')


@app.route('/upload_pdf', methods=['GET', 'POST'])
def upload_pdf():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename.endswith('.pdf'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            session['file_name'] = filename

            # Extract text from PDF and save it as 'extracted_text.txt'
            extracted_texts = convert_pdf_to_text(filepath)
            with open('report.txt', 'w') as f:
                for text in extracted_texts:
                    cleaned_text = clean_extracted_text(text)
                    f.write(cleaned_text + "\n")

            # Redirect to main query page after processing
            # return redirect(url_for('index'))
            return jsonify({'status': 'success'})

    return render_template('upload_pdf.html')

@app.route('/upload_image', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            session['file_name'] = filename

            extract_image(filepath)

            return redirect(url_for('index'))

    return render_template('upload_image.html')



if __name__ == '__main__':
    app.run(host="0.0.0.0")
