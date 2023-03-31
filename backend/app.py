from flask import Flask
from flask import render_template
from flask import request
from markupsafe import escape
import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__)


@app.route('/')
def hey():
    return render_template("base.html")


# DISCLAIMER
# This code is only an example and is subject to be replaced/removed.
# Feel free to use it as inspiration while it is still alive.
@app.route('/plot', methods=['GET', 'POST'])
def plot():
    plt.clf()

    def f(t, aa, bb): return np.exp(-t) * np.cos(aa * np.pi * t) - bb

    a = request.args.get('a', type=int)
    if not a: a = 0
    b = request.args.get('b', type=int)
    if not b: b = 0
    t1 = np.arange(0.0, 10.0, 0.01)
    col = request.args.get('mycolor')
    if not col: col = 'b'
    plt.plot(t1, f(t1, a, b), col)
    plt.savefig("./static/myplot.png")
    return render_template("plot.html")


if __name__ == '__main__':
    app.run()
