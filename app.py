

# This is basically the heart of my flask 


from flask import Flask, render_template, request, redirect, url_for, jsonify
import pickle
from model import ProductRecommendation

recommend = ProductRecommendation() ## ProductRecommendation is a class from the Model.py file
app = Flask(__name__)  

import os
from flask import send_from_directory

@app.route('/', methods = ['POST', 'GET'])
def home():
    flag = False 
    data = ""
    return render_template('index.html', data=data, flag=flag)

@app.route('/productList', methods = ['GET'])  ## product list from index.html is called
def productList():
    user=request.args.get("userid")
    data=recommend.getTop5Products(user)  ## calling the method for top 5 products
    # check if there was an error or did it return the list.
    if data.startswith("ERROR") :    ### checking if the user is not in db. ERROR was we set in model.py
        print(data + "Error") ## Print the error for futher debugging
        return render_template('index.html', error=data)
    return data

if __name__ == '__main__' :
    app.run(debug=False )  

