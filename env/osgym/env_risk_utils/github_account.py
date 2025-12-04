from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import requests
import re
import random
import time

app = Flask(__name__)

''''''
@app.route("/")
def index():
    return render_template("github_account.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5007, debug=True)