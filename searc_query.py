from flask import Flask, render_template, request, jsonify
import functions
import warnings
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore", category=FutureWarning, message="resume_download is deprecated*")
conversation_history = []

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query_text = request.form.get('query', '').strip()
        if not query_text:
            return jsonify({"bot_reply": "Please enter a valid query."}), 400

        try:
            structured_answers, pdf_filenames, pdf_descriptions = functions.process_query_clickhouse_pdf(query_text)
        except Exception as e:
            return jsonify({"bot_reply": f"An error occurred: {str(e)}"}), 500

        if not structured_answers:
            structured_answer = "I'm sorry, I couldn't find a relevant response for your query."
            pdf_filenames = []
            pdf_descriptions = []
        else:
            structured_answer = structured_answers[0]

        conversation_history.append((query_text, structured_answer, pdf_filenames, pdf_descriptions))
        return jsonify({
            "bot_reply": structured_answer,
            "pdf_urls": pdf_filenames,
            "pdf_descriptions": pdf_descriptions
        })

    return render_template('index.html', zip=zip, conversation_history=conversation_history)


@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)