from flask import Flask, render_template, request
from p1 import start_identification


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        m = request.values['m']
        start_identification(m)
        return render_template('Tq.html')

    return render_template('index.html')


if __name__ == '__main__':
    app.run()
