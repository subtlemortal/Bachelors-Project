# import the necessary packages
from flask import Flask, render_template, redirect, url_for, request,session,Response
from werkzeug import secure_filename
from supportFile import *
import os
import cv2
import pandas as pd
import utils
import nltk

app = Flask(__name__)

app.secret_key = '1234'
app.config["CACHE_TYPE"] = "null"
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route('/', methods=['GET', 'POST'])
def landing():
	return render_template('home.html')

@app.route('/home', methods=['GET', 'POST'])
def home():
	return render_template('home.html')

@app.route('/input', methods=['GET', 'POST'])
def input():
	if request.method == 'POST':
		if request.form['sub']=='Submit':
			num = request.form['num']
			users = {'Name':request.form['name'],'Email':request.form['email'],'Contact':request.form['num']}
			df = pd.DataFrame(users,index=[0])
			df.to_csv('users.csv',mode='a',header=False)

			sec = {'num':num}
			df = pd.DataFrame(sec,index=[0])
			df.to_csv('secrets.csv')

			return redirect(url_for('video'))

	return render_template('input.html')


@app.route('/video', methods=['GET', 'POST'])
def video():
	return render_template('video.html')

@app.route('/video_stream')
def video_stream():
	 return Response(get_frame(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/textmining',methods=['GET', 'POST'])
def textmining():
	if request.method == 'POST':
		username = request.form["name"]
		email = request.form["email"]
		num = request.form["num"]
		symptoms = request.form["symptoms"]
		print(username)
		print(email)
		print(symptoms)

		# define punctuation
		punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

		my_str = symptoms

		# To take input from the user
		# my_str = input("Enter a string: ")

		# remove punctuation from the string
		no_punct = ""
		for char in my_str:
			if char not in punctuations:
				no_punct = no_punct + char
		
		symptoms = no_punct
   
		
		utils.export("data/"+username+"-symptoms.txt", symptoms, "w")
				
		data = utils.getTrainData()

		def get_words_in_tweets(tweets):	
			all_words = []
			for (words, sentiment) in tweets:
	  			all_words.extend(words)
			return all_words

		def get_word_features(wordlist):		
		
			wordlist = nltk.FreqDist(wordlist)
			word_features = wordlist.keys()
			return word_features

		word_features = get_word_features(get_words_in_tweets(data))		
		


		def extract_features(document):		
			document_words = set(document)
			features = {}
			for word in word_features:
				#features[word.decode("utf8")] = (word in document_words)
				features[word] = (word in document_words)
			#print(features)
			return features

		allsetlength = len(data)
		print(allsetlength)		
		#training_set = nltk.classify.apply_features(extract_features, data[:allsetlength/10*8])		
		training_set = nltk.classify.apply_features(extract_features, data)
		#test_set = data[allsetlength/10*8:]		
		test_set = data[88:]		
		classifier = nltk.NaiveBayesClassifier.train(training_set)			
		
		def classify(symptoms):
			return(classifier.classify(extract_features(symptoms.split())))
			
				
			
		f = open("data/"+ username+"-symptoms.txt", "r")	
		f = [line for line in f if line.strip() != ""]	
		tot=0
		pos=0
		neg=0
		for symptom in f:
			tot = tot + 1
			result = classify(symptom)
			if(result == "Depression Detected"):
				neg = neg + 1
			print(result)
	
		pos = tot - neg
		if(neg > pos):
			result = "Depression Detected: " + str((neg/tot)*100) + "%"

			'''
			message = client.messages \
									.create(
											body = "https://www.youtube.com/watch?v=2UtwSI7lgkQ",
											from_='+14696544981',
											to="+91"+str(num)
										)
			'''
		else:
			result = "No Depression Detected"


		return render_template('textmining.html',name=username,num = num,email=email,symptoms=symptoms,result=result)			    
	return render_template('textmining.html')

# No caching at all for API endpoints.
@app.after_request
def add_header(response):
	# response.cache_control.no_store = True
	response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
	response.headers['Pragma'] = 'no-cache'
	response.headers['Expires'] = '-1'
	return response


if __name__ == '__main__':
	app.run(host='0.0.0.0', debug=True, threaded=True)
