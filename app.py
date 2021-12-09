from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
import numpy as np
import tensorflow as tf

### LOAD KERAS MODEL ###
# sess = tf.compat.v1.Session()
graph = tf.compat.v1.get_default_graph()

# with sess.as_default():
#with graph.as_default():
    #heart = tf.keras.models.load_model("my_model.h5")

app = Flask(__name__)

# Apply Flask CORS
CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"

@app.route('/', methods=['POST', 'GET'])
def get_data():
    return render_template("index.html")

@app.route('/benhtim', methods=["POST", "GET"])
def benhtim():
    rq = request.form
    
    age = [request.form["age"]]
    gender = [int(request.form["gender"])]
    cpt_list = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    cpt = cpt_list[int(request.form["cpt"])]
    rbp = [request.form["restingbp"]]
    cholesterol = [request.form["cholesterol"]]
    fbs = [int(request.form["fbs"])]
    recg_list = [[0, 0], [1, 0], [0, 1]]
    restingecg = recg_list[int(request.form["restingecg"])]
    maxHR = [request.form["maxhr"]]
    exangina = [int(request.form["exAngina"])]
    oldpeak = [float(request.form["oldpeak"])]
    stslope_list = [[0, 0], [1, 0], [0, 1]]
    stslope = stslope_list[int(request.form["stslope"])]
    
    data = age + rbp + cholesterol + fbs + maxHR + oldpeak + cpt + gender + restingecg + exangina + stslope
    data = np.reshape(data, (1, 15))
    
    #from sklearn.preprocessing import StandardScaler
    #sc = StandardScaler()
    #data = sc.fit_transform(data)
    with graph.as_default():
        heart = tf.keras.models.load_model("my_model.h5")
        pred = heart.predict(data)
    pred = pred[0][0]
    pred=0 if pred<0.5 else 1
    
    return render_template("after.html", data=pred)


if __name__ == '__main__':
    app.run(port=12000)
    


