from flask import Flask, render_template, request
from text_extraction import extract_text_from_url

# importing the saved model
from model import clean_text, classifier, vectorization
from joblib import dump, load
dump(classifier, 'model.joblib')
loaded_model=load('model.joblib')





app=Flask(__name__)



@app.route('/', methods=['GET','POST'])
def index():
    if request.method=='POST':
        url = request.form['url']
        text = extract_text_from_url(url)
        cleaned_text = clean_text(text)
        text_vector = vectorization.transform([cleaned_text])
        predicted_label = classifier.predict(text_vector)
        prediction=predicted_label[0]
        return render_template('result.html', prediction=prediction)
    
    return render_template('index.html')
    
  

if __name__=='__main__':
    app.run(debug=True)