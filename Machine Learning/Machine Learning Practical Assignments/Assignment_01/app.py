from flask import Flask, request, render_template

from prediction_value.predValue import Prediction


app = Flask(__name__)


@app.route("/", methods=['GET'])
def home():
    return render_template('index.html')



@app.route("/predict", methods=['POST'])
def predictPrice():
    if request.form is not None:
      form_data = request.form
      loadvalObj= Prediction(form_data)
      y_pred = loadvalObj.predictValue()

      return render_template('prediction.html',value=y_pred)




if __name__ == '__main__':
    app.run(debug=True)