from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.metrics import AUC
import numpy as np

app = Flask(__name__)

dependencies = {
    'auc_roc': AUC
}

verbose_name = {
0: 'Myocardial_Infarction',
1: 'History_of_Myocardial_Infarction', 
2: 'Abnormal_Heartbeat',
3: 'Normal_person',
 
 
 
           }



model = load_model('ecg.h5')

def predict_label(img_path):
	test_image = image.load_img(img_path, target_size=(224,224))
	test_image = image.img_to_array(test_image)/255.0
	test_image = test_image.reshape(1, 224,224,3)

	predict_x=model.predict(test_image) 
	classes_x=np.argmax(predict_x,axis=1)
	
	return verbose_name[classes_x[0]]

 
@app.route("/")
@app.route("/first")
def first():
	return render_template('first.html')
    
@app.route("/login")
def login():
	return render_template('login.html')   
    
@app.route("/index", methods=['GET', 'POST'])
def index():
	return render_template("index.html")


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/tests/" + img.filename	
		img.save(img_path)

		predict_result = predict_label(img_path)

	return render_template("prediction.html", prediction = predict_result, img_path = img_path)

@app.route("/performance")
def performance():
	return render_template('performance.html')
    
@app.route("/chart")
def chart():
	return render_template('chart.html') 

	
if __name__ =='__main__':
	app.run(debug = True)




# from flask import Flask, render_template, request
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.metrics import AUC
# import numpy as np
# import os

# app = Flask(__name__)

# # Custom object needed for loading the model
# dependencies = {
#     'AUC': AUC
# }

# verbose_name = {
#     0: 'Myocardial_Infarction',
#     1: 'History_of_Myocardial_Infarction', 
#     2: 'Abnormal_Heartbeat',
#     3: 'Normal_person',
# }

# # Load the model with custom objects
# model_path = 'ecg.h5'
# try:
#     model = load_model(model_path, custom_objects=dependencies)
# except Exception as e:
#     model = None
#     print(f"Error loading model: {e}")

# def predict_label(img_path):
#     test_image = image.load_img(img_path, target_size=(224, 224))
#     test_image = image.img_to_array(test_image) / 255.0
#     test_image = np.expand_dims(test_image, axis=0)

#     predict_x = model.predict(test_image) 
#     classes_x = np.argmax(predict_x, axis=1)
    
#     return verbose_name.get(classes_x[0], "Unknown")

# @app.route("/")
# @app.route("/first")
# def first():
#     return render_template('first.html')
    
# # @app.route("/login")
# # def login():
# #     return render_template('login.html')   
    
# @app.route("/index", methods=['GET', 'POST'])
# def index():
#     return render_template("index.html")

# @app.route("/submit", methods=['GET', 'POST'])
# def get_output():
#     if request.method == 'POST':
#         img = request.files['my_image']

#         # Ensure the static/tests directory exists
#         img_dir = "static/tests/"
#         if not os.path.exists(img_dir):
#             os.makedirs(img_dir)

#         img_path = os.path.join(img_dir, img.filename)    
#         img.save(img_path)

#         predict_result = predict_label(img_path)

#         return render_template("prediction.html", prediction=predict_result, img_path=img_path)

# @app.route("/performance")
# def performance():
#     return render_template('performance.html')
    
# @app.route("/chart")
# def chart():
#     return render_template('chart.html') 

# if __name__ == '__main__':
#     app.run(debug=True)
# from flask import Flask, render_template, request
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.metrics import AUC
# import numpy as np
# import os

# app = Flask(__name__)

# # Custom object needed for loading the model
# dependencies = {
#     'AUC': AUC
# }

# verbose_name = {
#     0: 'Myocardial_Infarction',
#     1: 'History_of_Myocardial_Infarction', 
#     2: 'Abnormal_Heartbeat',
#     3: 'Normal_person',
# }

# # Load the model with custom objects
# model_path = 'ecg.h5'
# try:
#     model = load_model(model_path, custom_objects=dependencies)
# except Exception as e:
#     model = None
#     print(f"Error loading model: {e}")

# def predict_label(img_path):
#     if model is None:
#         return "Model not loaded. Cannot make predictions."
    
#     test_image = image.load_img(img_path, target_size=(224, 224))
#     test_image = image.img_to_array(test_image) / 255.0
#     test_image = np.expand_dims(test_image, axis=0)

#     predict_x = model.predict(test_image) 
#     classes_x = np.argmax(predict_x, axis=1)
    
#     return verbose_name.get(classes_x[0], "Unknown")

# @app.route("/")
# @app.route("/first")
# def first():
#     return render_template('first.html')
    
# @app.route("/login")
# def login():
#     return render_template('login.html')   
    
# @app.route("/index", methods=['GET', 'POST'])
# def index():
#     return render_template("index.html")

# @app.route("/submit", methods=['GET', 'POST'])
# def get_output():
#     if request.method == 'POST':
#         img = request.files['my_image']

#         # Ensure the static/tests directory exists
#         img_dir = "static/tests/"
#         if not os.path.exists(img_dir):
#             os.makedirs(img_dir)

#         img_path = os.path.join(img_dir, img.filename)    
#         img.save(img_path)

#         predict_result = predict_label(img_path)

#         return render_template("prediction.html", prediction=predict_result, img_path=img_path)

# @app.route("/performance")
# def performance():
#     return render_template('performance.html')
    
# @app.route("/chart")
# def chart():
#     return render_template('chart.html') 

# if __name__ == '__main__':
#     app.run(debug=True)



	

	


