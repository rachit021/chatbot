from flask import Flask, request, render_template
from bot import chatbot

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['user_input']
        bot_response = chatbot(user_input)
        return render_template('index.html', bot_response=bot_response)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)