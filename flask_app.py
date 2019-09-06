from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug import secure_filename
import os, json

UPLOAD_FOLDER = './upload'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

@app.route("/")
def index():
   return render_template("index.html")


@app.route('/upload')
def upload_file():
   return render_template('upload.html')


@app.route('/test/', methods=['GET', 'POST'])
def test():
     clicked = None
     if request.method == "POST":
          clicked = request.json['data']
     return render_template('test.html')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/success', methods= ['GET','POST'])
def success():
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'],f.filename))
        print(f.filename)
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        return json.dumps({'filename': full_filename})
        # return 

# @app.route('/upload', methods=['GET', 'POST'])
# def uploader():


# #    if request.method == 'POST':
# #       f = request.files['file']
# #       f.save(secure_filename(f.filename))
# #       return 'file uploaded successfully'
#     if request.method == 'POST':
#         # check if the post request has the file part
#         if 'file' not in request.files:
#             flash('No file part')
#             return redirect(request.url)
#         file = request.files['file']
#         # if user does not select file, browser also
#         # submit an empty part without filename
#         if file.filename == '':
#             flash('No selected file')
#             return redirect(request.url)
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#             return redirect(url_for('uploaded_file',
#                                     filename=filename))
#     return render_template("index.html")
    

if __name__ == '__main__':
   app.run(debug=True)
