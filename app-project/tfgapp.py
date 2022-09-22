from flask import Flask ,render_template,request,url_for,redirect,flash,session,g
from flask_restful import reqparse, abort, Api, Resource
from algoritmo import test
import shutil

app = Flask(__name__)
app.secret_key = "super secret key"


@app.route('/', methods = ['GET'])
def index():
    return render_template('index.html')

@app.route('/' , methods = ['POST'])
def ejecutar():
    imagen = request.files['file']
    image_path = 'data/imagenes-test/' + imagen.filename
    shutil.copy(image_path, "static/images")
    image_path_new = "static/images/" + imagen.filename
    patches_path = 'data/patches-test'
    res = test(patches_path, image_path)
    return render_template('results.html', res = res, imagen = image_path_new)

if __name__ == '__main__':
    app.run(debug=True)