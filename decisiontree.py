from flask import Flask, request, jsonify
from pymongo import MongoClient
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Kết nối MongoDB
client = MongoClient('mongodb+srv://sangvo22026526:5anG15122003@cluster0.rcd65hj.mongodb.net/anime_tango2')
db = client['anime_tango2']
anime_collection = db['Anime']
user_rating_collection = db['UserRating']

# Flask app
app = Flask(__name__)

# Hàm lấy dữ liệu Anime
def get_anime_data():
    anime_data = list(anime_collection.find())
    return pd.DataFrame(anime_data)

# Hàm lấy dữ liệu UserRatings
def get_user_ratings(user_id):
    user_ratings = list(user_rating_collection.find({'User_id': user_id}))
    return user_ratings

# Xử lý dữ liệu Anime
anime_df = get_anime_data()
anime_df2 = anime_df.copy()

# Thêm các cột xử lý
def categorize_score(score):
    if score < 8:
        return 0
    elif 8 <= score <= 9:
        return 1
    else:
        return 2

anime_df['Score_'] = anime_df['Score'].apply(categorize_score)
genres = ['Action', 'Adventure', 'Comedy', 'Drama', 'Romance']  # Cập nhật danh sách Genres
for genre in genres:
    anime_df[genre] = anime_df['Genres'].apply(lambda x: 1 if genre in x else 0)

anime_df['Favorites_'] = anime_df['Favorites'].apply(lambda x: 0 if x <= 5000 else (1 if x <= 20000 else 2))
anime_df['JapaneseLevel_'] = anime_df['JapaneseLevel'].apply(lambda x: 0 if x in ['N4', 'N5'] else (1 if x in ['N2', 'N3'] else 2))
anime_df['AgeCategory'] = anime_df['Old'].apply(lambda x: 1 if '13+' in x else 0)

# Tạo đặc trưng người dùng
def get_user_features(user_id):
    user_ratings = get_user_ratings(user_id)
    user_ratings_df = pd.DataFrame(user_ratings)
    user_anime_df = anime_df[anime_df['Anime_id'].isin(user_ratings_df['Anime_id'])]

    features = {}
    features['Avg_Old'] = user_anime_df['AgeCategory'].mean()
    features['Avg_Favorites'] = user_anime_df['Favorites_'].mean()
    features['Avg_JapaneseLevel'] = user_anime_df['JapaneseLevel_'].mean()
    features['Avg_Score'] = user_anime_df['Score_'].mean()
    for genre in genres:
        features[f'Avg_{genre}'] = user_anime_df[genre].mean()

    return features

# Huấn luyện mô hình Decision Tree
def train_decision_tree(user_id):
    user_features = get_user_features(user_id)
    anime_features = anime_df[genres + ['Favorites_', 'JapaneseLevel_', 'AgeCategory', 'Score_']]
    user_feature_vector = np.array([user_features[f'Avg_{genre}'] for genre in genres] +
                                   [user_features['Avg_Favorites'], user_features['Avg_JapaneseLevel'],
                                    user_features['Avg_Old'], user_features['Avg_Score']])

    X = anime_features
    y = anime_df['Score_']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    return clf

# Gợi ý Anime
def recommend_anime_using_tree(user_id, n):
    clf = train_decision_tree(user_id)
    anime_features = anime_df[genres + ['Favorites_', 'JapaneseLevel_', 'AgeCategory', 'Score_']]
    predictions = clf.predict(anime_features)

    recommended_anime_indices = np.where(predictions >= 1)[0]
    recommended_anime = anime_df2.iloc[recommended_anime_indices]

    user_ratings = get_user_ratings(user_id)
    rated_anime_ids = [rating['Anime_id'] for rating in user_ratings]
    recommended_anime = recommended_anime[~recommended_anime['Anime_id'].isin(rated_anime_ids)]
    recommended_anime = recommended_anime.head(n)[['Anime_id', 'Name', 'Score', 'Genres']]

    return recommended_anime

# API endpoint
@app.route('/api/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_id = data.get('user_id')
    n = data.get('n', 5)

    if not user_id:
        return jsonify({'error': 'user_id is required'}), 400

    recommended_anime = recommend_anime_using_tree(user_id, n)
    return jsonify(recommended_anime.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
