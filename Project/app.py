from flask import Flask,redirect,url_for,render_template





app = Flask(__name__,template_folder='template')


@app.route("/")
def home():
    return render_template("index.html")





app.run(host="localhost",port=80,debug=True)