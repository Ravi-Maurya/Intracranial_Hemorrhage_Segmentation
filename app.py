from flask import Flask, render_template, request, redirect, send_from_directory, url_for
from werkzeug import secure_filename
import model
import os

basedir = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(basedir, 'images')


@app.route('/')
def homepage():
    return render_template("home.html")


@app.route('/uploads', methods=['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        directory = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(directory):
            return redirect(url_for('run_model', filename=filename))
        file.save(directory)
        return redirect(url_for('run_model', filename=filename))
    return render_template("home.html")


@app.route('/<filename>')
def run_model(filename):
    nn = model.DeepModel(batch_size=8)
    nn.predict(filename)
    res = nn.result()
    nn.destroy()
    del nn
    return render_template("prediction.html", res=res)
    # return send_from_directory(app.config['UPLOAD_FOLDER'],filename)


if __name__ == '__main__':
    app.run(debug=True)
