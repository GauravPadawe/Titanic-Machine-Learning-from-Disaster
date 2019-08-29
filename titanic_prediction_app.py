### Here we will import numpy , flask and pickle
import numpy as np
from flask import Flask, render_template, request
from pickle import load

### Let us Initialize the Flask app
app = Flask(__name__)
### pickle model extraction 
model_load = load(open('titanic_model.pkl', 'rb'))


### CODE BLOCK 1
### this line of code will route us to default page
@app.route('/')
### user-defined function w.r.t implentation of render_template
### render_template re-directs us to home page by default and after every execution
def home():
    return render_template('index.html')
### END OF CODE BLOCK 1


### CODE BLOCK 2
### this section will route or create a additional page predict
### with method post it will read feature inputs we provide
@app.route('/predict', methods=['POST'])
### user-defined function to predict
def predict():
	### will request the values from form
    int_features = [int(x) for x in request.form.values()]
	### Forming a array
    final_features = [np.array(int_features)]
    ### Predict on input that is passed
    prediction = model_load.predict(final_features)[0]
    ### If else condition with implementation of home page for rendering after presenting result
    if prediction == 1:
        return render_template('index.html', prediction_text='Subject Survived')
    else:
        return render_template('index.html', prediction_text='Subject Died')
### END OF CODE BLOCK 2

### debugging the app
if __name__=="__main__":
    app.run(debug=True)
