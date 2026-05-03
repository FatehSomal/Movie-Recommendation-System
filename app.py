from flask import Flask, render_template, request
from Backend import recommend, df1

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("frontend.html")

@app.route("/search")
def search():
    query = request.args.get("q")

    if not query:
        return {"results": []}
    
    matches = df1[df1['title'].str.lower().str.contains(query.lower())]

    titles = matches['title'].head(10).tolist()

    return {"results": titles}

@app.route("/recommended", methods=["post"])
def get_recommendations():

    movie_name = request.form["movie"]

    results = recommend(movie_name)

    return render_template("frontend.html", recommendations=results)

if __name__ == '__main__':
    app.run(debug=True)