import pickle

with open("BERT/processed_scaled_movies.pkl", 'rb') as file:
    df = pickle.load(file)

with open("BERT/processed_movies.pkl", 'rb') as file:
    df1 = pickle.load(file)

with open("BERT/movie_embeddings.pkl", 'rb') as file:
    embeddings = pickle.load(file)

with open("BERT/nn_model.pkl", 'rb') as file:
    nn = pickle.load(file)



def recommend(movie_name, top_n=10):
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

    for i, score in results[:top_n]:
        output.append({
            'title': df1.iloc[i]['title'],
            'rating': round(df1.iloc[i]['avg_rating'], 3),
            'year': df1.iloc[i]['year'],
            'genres': df1.iloc[i]['genres'],
            'runtime': int(df1.iloc[i]['runtime']),
            'director': df1.iloc[i]['cast_clean'],
            'overview': df1.iloc[i]['overview']
        })

    return (output)