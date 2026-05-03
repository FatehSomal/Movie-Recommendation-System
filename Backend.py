import pickle
import requests

API_KEY = "889054f26c198c3f3362ecd5cbdb88d1"

with open("BERT/processed_scaled_movies.pkl", 'rb') as file:
    df = pickle.load(file)

with open("BERT/processed_movies.pkl", 'rb') as file:
    df1 = pickle.load(file)

with open("BERT/movie_embeddings.pkl", 'rb') as file:
    embeddings = pickle.load(file)

with open("BERT/nn_model.pkl", 'rb') as file:
    nn = pickle.load(file)

poster_cache = {}


def get_poster(title, year=None):

    key = f"{title}_{year}"

    if key in poster_cache:
        return poster_cache[key]

    url = "https://api.themoviedb.org/3/search/movie"

    params = {
        "api_key": API_KEY,
        "query": title
    }

    if year:
        params["year"] = year

    try:
        response = requests.get(url, params=params, timeout=5)
        data = response.json()

        if data.get("results"):
            poster_path = data["results"][0].get("poster_path")

            if poster_path:
                poster_url = "http://image.tmdb.org/t/p/w500" + poster_path

                poster_cache[key] = poster_url

                return poster_url
        
    except Exception as e:
        print("Error fetching poster:", e)

    fallback = "https://via.placeholder.com/150"
    poster_cache[key] = fallback

    return fallback



def recommend(movie_name, top_n=5):
    movie_name = movie_name.strip()
    
    matches = df[df['title'].str.lower() == movie_name.lower()]

    if len(matches) == 0:
        return []

    idx = matches.index[0]

    query_vector = embeddings[idx].reshape(1, -1)

    distances, indicies = nn.kneighbors(query_vector)

    query_genres = set(str(df.iloc[idx]['genres']).split())

    results = []

    for rank, i in enumerate(indicies[0]):

        if i == idx:
            continue

        sim = 1 - distances[0][rank]

        candidate_genres = set(str(df.iloc[i]['genres']).split())
        genre_score = len(query_genres.intersection(candidate_genres))
        genre_score = min(genre_score / 3, 1)

        year_diff = abs(df.iloc[i]['year'] - df.iloc[idx]['year'])
        year_score = max(0, 1 - year_diff / 10)

        final_score = (
            0.55 * sim +
            0.15 * genre_score +
            0.10 * df.iloc[i]['avg_rating'] +
            0.10 * df.iloc[i]['popularity'] +
            0.05 * df.iloc[i]['vote_count'] +
            0.05 * year_score
        )

        results.append((i, final_score))

    results = sorted(results, key=lambda x: x[1], reverse=True)

    output = []

    for idx, score in results[:top_n]:

        idx = int(idx)

        row = df1.iloc[idx]

        cast_words = str(row['cast_clean']).split()
        cast_pairs = [" ".join(cast_words[j:j+2]) for j in range(0, len(cast_words), 2)]
        top_cast = ", ".join(cast_pairs[:5])

        title = row['title']
        year = int(row['year'])
        poster = get_poster(title, year)

        rating = row['avg_rating']

        rounded_stars = round(rating)
        empty_stars = 5 - rounded_stars

        genres = ", ".join(str(row['genres_clean']).split())

        output.append({
            'title': title,
            'rating': round(rating, 1),
            'rounded_stars': rounded_stars,
            'empty_stars': empty_stars,
            'year': year,
            'genres': genres,
            'runtime': int(row['runtime']),
            'director': row['director_clean'],
            'cast': top_cast,
            'overview': row['overview'],
            'poster': poster
        })

    return (output)