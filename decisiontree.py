from fastapi import FastAPI, Request
from pymongo import MongoClient
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

# Thêm middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


client = MongoClient("mongodb+srv://sangvo22026526:5anG15122003@cluster0.rcd65hj.mongodb.net/anime_tango2")
db = client["anime_tango2"]

# Tải dữ liệu từ MongoDB
df_anime = pd.DataFrame(list(db["Anime"].find()))
df_favorites = pd.DataFrame(list(db["UserFavorites"].find()))

# Chuyển dữ liệu _id thành chuỗi (nếu cần)
df_anime['_id'] = df_anime['_id'].astype(str)
df_anime['Anime_id'] = df_anime['Anime_id'].astype(str)
df_favorites['_id'] = df_favorites['_id'].astype(str)

# Tiền xử lý dữ liệu:

# 1. Phân loại Members vào các mốc: 0-200000, 200000-500000, và 500000+
def categorize_members(members):
    if members <= 200000:
        return '0-200000'
    elif members <= 500000:
        return '200000-500000'
    else:
        return '500000+'

df_anime['Members_Category'] = df_anime['Members'].apply(categorize_members)

# 2. Mã hóa cột JapaneseLevel
le_japanese_level = LabelEncoder()
df_anime['JapaneseLevel'] = le_japanese_level.fit_transform(df_anime['JapaneseLevel'])

# 3. Mã hóa Genres thành các cột riêng biệt (One-hot Encoding)
df_anime = pd.concat([df_anime, df_anime['Genres'].apply(pd.Series)], axis=1)
df_anime.drop('Genres', axis=1, inplace=True)

# 4. Chuyển các dữ liệu phân loại thành dạng số (numerical) nếu cần
le = LabelEncoder()
df_anime['Type'] = le.fit_transform(df_anime['Type'])

# Lấy ra các thông tin Anime mà người dùng đã yêu thích từ bảng UserFavorites
favorite_animes = df_favorites[['User_id', 'favorites']]

# Tạo DataFrame kết hợp các bộ phim yêu thích của người dùng
favorite_data = df_anime[df_anime['Anime_id'].isin(favorite_animes['favorites'])]

# Đặc trưng đầu vào (features) cho mô hình: Loại bỏ 'Status' và 'Producers', thêm 'Members_Category' và 'JapaneseLevel'
features = favorite_data[['Score', 'Type', 'Members_Category', 'JapaneseLevel'] + [col for col in df_anime.columns if col not in ['_id', 'Anime_id', 'Name', 'English_Name', 'Favorites', 'Scored_By', 'Member', 'Image_URL', 'JapaneseLevel', 'LastestEpisodeAired']]]
target = favorite_data['Anime_id']  # Mục tiêu là gợi ý Anime_id cho người dùng

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Khởi tạo và huấn luyện mô hình Decision Tree
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Đánh giá mô hình
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy}")

# Hàm gợi ý phim cho người dùng
@app.post("/")
async def recommend(request: Request):
    data = await request.json()
    user_id = data.get("user_id")
    n = data.get("n", 10)  # Số lượng gợi ý, mặc định là 10
    # Lấy các bộ phim yêu thích của người dùng từ bảng UserFavorites
    user_favorites = df_favorites[df_favorites['User_id'] == user_id]['favorites'].tolist()
    
    # Lọc ra các bộ phim chưa được yêu thích (để tránh gợi ý lại phim đã yêu thích)
    potential_animes = df_anime[~df_anime['Anime_id'].isin(user_favorites)]
    
    # Dự đoán các phim mà người dùng có thể thích
    features = potential_animes[['Score', 'Type', 'Members_Category', 'JapaneseLevel'] + [col for col in df_anime.columns if col not in ['_id', 'Anime_id', 'Name', 'English_Name', 'Favorites', 'Scored_By', 'Member', 'Image_URL', 'JapaneseLevel', 'LastestEpisodeAired']]]
    predicted = clf.predict(features)
    
    # Lấy các Anime_id dự đoán từ mô hình
    recommended_animes = potential_animes[potential_animes['Anime_id'].isin(predicted)].head(n)  # Lấy top n gợi ý
    
    recommendations = recommended_animes[['Anime_id', 'Name', 'English_Name', 'Score', 'Genres']].to_dict(orient='records')
    return recommendations
