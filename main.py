from flask import Flask, request, jsonify
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
import pickle
import json

app = Flask(__name__)

data = pd.read_csv("../RecommendedSystem/data.csv")

tfidf_matrix = pickle.load(open('tfidf_matrix.pickle', 'rb'))


@app.route("/api/v1/recommend/", methods=["GET"])
def recommend():
    # user_input = request.get_json()
    # title = user_input['title']
    #
    test_data = data["название"].iloc[22]

    result = get_recommendations(test_data, tfidf_matrix)
    end_result = {
        "result": result
    }
    return json.dumps(end_result, indent=4, ensure_ascii=False, ).encode("utf8")


def get_recommendations(title, tfidf_matrix):
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(data.index, index=data['название']).drop_duplicates()
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    topMovies = data['название'].iloc[movie_indices]
    movie_ratings = [i[1] for i in sim_scores]
    result = []
    for i in range(len(topMovies)):
        anime = {}
        anime["name"] = topMovies.iloc[i]
        anime["score"] = movie_ratings[i]
        result.append(anime)
    return result


if __name__ == "__main__":
    app.run()
