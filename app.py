from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/test')
def hello():
    return {'test1': 1, 'test2': 2}