from flask import Flask, render_template, jsonify
import json
import os

app = Flask(__name__, template_folder='templates')

progress_file = 'progress.json'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/progress')
def progress():
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    else:
        return jsonify({"message": "No progress data available."})

if __name__ == '__main__':
    app.run(debug=True)