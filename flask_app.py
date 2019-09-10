from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug import secure_filename
import os, json
import os.path
from os import path
import paint_by_num

UPLOAD_FOLDER = './static/upload'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


colors = detail_level = full_filename = ""
colored_output = color_palette = outline_image_with_no = ""

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


@app.route('/success', methods= ['POST'])
def success():
    global colors
    global detail_level
    global full_filename
    global colored_output
    global color_palette
    global outline_image_with_no
    if request.method == 'POST':
        # os.remove(file) for file in os.listdir('./static/processed_image') if file.endswith('.jpg')
        for filename in os.listdir('./static/processed_image'):
            if filename.endswith('.jpg') or filename.endswith('.pdf'):
                if path.isfile('./static/processed_image/' + filename):
                    os.remove('./static/processed_image/' + filename)
                    print(filename)
        print(request.form)
        f = request.files['file']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'],f.filename))
        print(f.filename)
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        colors = request.form['color_slider']
        detail_level = request.form['detail_slider']
        # return json.dumps({'filename': str(full_filename), 'colors':int(colors), 'detail':int(detail_level)})
        print(str(full_filename) +' '+ str(int(colors)) + str(int(detail_level)-1))
        colored_output, color_palette, outline_image_with_no,output_pdf = paint_by_num.processImage(str(full_filename),int(colors),int(detail_level)-1)
        return render_template('success.html', o_img=os.path.basename(full_filename), c_img=os.path.basename(colored_output), ot_img=os.path.basename(outline_image_with_no), cp=os.path.basename(color_palette), o_pdf=os.path.basename(output_pdf))


@app.route('/get_data', methods= ['GET'])
def getData():
    global colors
    global detail_level
    global full_filename
    global colored_output
    global color_palette
    global outline_image_with_no
    if request.method == "GET":
        if len(str(full_filename)) > 1:
            # paint_by_num.processImage(str(full_filename),int(colors),int(detail_level))
            return json.dumps({'orig_image': str(full_filename), 'colors':int(colors), 'detail':int(detail_level),'colored_image':str(colored_output),'palette':str(color_palette),'outline_image':str(outline_image_with_no)})
        else:
            return "No data received!"

@app.route('/get_images',methods= ['GET'])
def getImages():
    global colors
    global detail_level
    global full_filename
    global colored_output
    global color_palette
    global outline_image_with_no
    



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
