from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import pickle

# load the model from disk
filename = 'review_score_classifier.pkl'
clf = pickle.load(open(filename, 'rb'))
tfidf=pickle.load(open('transform.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
#   reviews = pd.read_csv("Amazon_Reviews_New.csv")
#   reviews = reviews[['Score', 'Text_STEM_CLEAN']]
#   reviews_sample = reviews.sample(n=100000, replace=False, random_state=42)
#   tfidf_v=TfidfVectorizer(max_features=5000,ngram_range=(1,3))
#   X=tfidf_v.fit_transform(reviews_sample['Text_STEM_CLEAN']).toarray()
#   pickle.dump(tfidf_v, open('transform.pkl', 'wb'))
#   classifier=LogisticRegression(class_weight='Balanced', max_iter=200)
#   classifier.fit(X_train, y_train)
#   pred = classifier.predict(X_test)


	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = tfidf.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)