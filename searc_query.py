from flask import Flask, render_template, request
import functions
import warnings

# Suppress the specific FutureWarning from Hugging Face Hub



app = Flask(__name__)

warnings.filterwarnings("ignore", category=FutureWarning, message="`resume_download` is deprecated*")
conversation_history = []
@app.route('/', methods=['GET', 'POST'])
def index():
    structured_sentence = None
    if request.method == 'POST':
        query_text = request.form['query']
        structured_sentence, pdf_filename = functions.process_query_clickhouse_word(query_text)
        conversation_history.append((query_text, structured_sentence, pdf_filename))
    return render_template('index.html', conversation_history=conversation_history)


@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=5000)
    app.run(debug=True)

