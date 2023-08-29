from flask import Flask, render_template, request, flash, redirect
from API import transfer_style
import os
import matplotlib.pyplot as plt

UPLOAD_FOLDER = './static/images/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

# We define a variable called app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# check if file extension is right
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# tells Flask what URL a user has to call the function below
# you will need to browse the url : '/test'
@app.route("/")
def home():
    return render_template("index.html")


@app.after_request
def set_response_headers(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# main flow of programme
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    try:
        # remove older files
        os.system("find static/images/ -maxdepth 1 -mmin +5 -type f -delete")
    except OSError:
        pass
    if request.method == 'POST':
        # check if the post request has the file part
        if 'content-file' and 'style-file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        content_file = request.files['content-file']
        style_file = request.files['style-file']
        files = [content_file, style_file]
        content_file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'content.jpg'))
        style_file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'style.jpg'))
        
        for i, file in enumerate(files):
            # if user does not select file, browser also
            # submit an empty part without filename
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            else:
                model_path = r"./model"
                img = transfer_style(content_file,style_file,model_path)
                plt.imsave(app.config['UPLOAD_FOLDER']+'stylized_image.jpg',img)
    return  render_template('sucess.html')
                

if __name__=='__main__':
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(debug = True)