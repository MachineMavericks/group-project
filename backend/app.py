# DEFAULT IMPORTS:
import os  # OS
import shutil  # SHUTIL
# FLASK IMPORTS:
from flask import Flask, render_template, request, flash  # FLASK
from werkzeug.utils import secure_filename
# CONTROLLERS=
from src.Controllers.NXGraphController import *  # NXGRAPH CONTROLLER

# PATHS=
input_dir = "static/input/"
output_dir = "static/output/"
os.mkdir(output_dir) and print("Can't find output directory. Creating one now.") \
    if not os.path.isdir(output_dir) else print("Found existing output directory.")
upload_dir = 'static/input/uploads'
allowed_extensions = {'csv', 'xlsx', 'tsv', 'ods'}

# FLASK APP:
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = upload_dir

# BLUEPRINTS/CONTROLLERS:
app.register_blueprint(nxgraph_bp)


# INDEX ROUTE:
@app.route('/', methods=['GET', 'POST'])
def index():
    """
    This function renders the template of the web app and redirect the user to the home page.
    :return:
    """
    return render_template("base.html")


# UPLOAD ROUTE:
def allowed_file(filename):
    """
    This function returns True/False if whether or not the file is allowed to be uploaded.
    :param filename: The path of the file to verify.
    :return: A boolean defining if whether or not the file is allowed.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


@app.route('/choice', methods=['GET', 'POST'])
def choose_dataset():
    """
    This function allows to render the templates to select the dataset, and renders a loading bar.
    :return: A rendered templates of the dataset selection.
    """
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return render_template("dataset/upload_not_valid.html")
        file = request.files['file']
        # If the user does not select a file, the page is refreshed.
        if file.filename == '':
            return render_template("dataset/upload_not_valid.html")
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return render_template("dataset/upload_done.html")
    return render_template("dataset/dataset_selection.html")

@app.route('/clear-cache', methods=['GET', 'POST'])
def clear_cache():
    """
    This function returns the base template used throughout the application.
    :return: The base template of the web-application.
    """
    shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    return render_template("base.html")


# MAIN:
def main():
    # START THE APP:
    app.run(debug=True)


if __name__ == '__main__':
    main()
