from flask import Flask, render_template, request
from Backend import recommend

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("frontend.html")

@app.route("/recommended", methods=["post"])
def get_recommendations():

    movie_name = request.form["movie"]

    results = recommend(movie_name)

    return render_template("frontend.html", recommendations=results)

if __name__ == '__main__':
    app.run(debug=True)