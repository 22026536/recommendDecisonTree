from fastapi import FastAPI, Request
import pandas as pd
from pymongo import MongoClient
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Khởi tạo app
app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

# Thêm middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả origin (hoặc chỉ định cụ thể như ["https://your-frontend.com"])
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = MongoClient("mongodb+srv://sangvo22026526:5anG15122003@cluster0.rcd65hj.mongodb.net/web_project")
db = client["anime_tango2"]

# Tải dữ liệu từ MongoDB
df_ratings = pd.DataFrame(list(db["UserRating"].find()))
df_anime = pd.DataFrame(list(db["Anime"].find()))


# Mã hóa các giá trị chuỗi thành số
label_encoder = LabelEncoder()
df_anime['JapaneseLevel'] = label_encoder.fit_transform(df_anime['JapaneseLevel'])

# Kết hợp thông tin xếp hạng vào dữ liệu phim
df_merged = df_ratings.merge(df_anime, on="Anime_id")

# Tạo trọng số cho cột JapaneseLevel (ví dụ: nhân với hệ số 2)
df_merged['Weighted_JapaneseLevel'] = df_merged['JapaneseLevel'] * 2

# Thêm "Weighted_JapaneseLevel" vào tập đặc trưng
X = df_merged[['User_id', 'Rating', 'Weighted_JapaneseLevel']]  # Đặc trưng đầu vào
y = df_merged['Name_encoded']                                   # Nhãn đầu ra

# Tách tập dữ liệu thành train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Mô hình Decision Tree
decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X_train, y_train)


def recommend_by_decision_tree(user_id, n):
    """
    Gợi ý phim sử dụng mô hình Decision Tree, trả về toàn bộ thông tin anime.
    """
    # Lọc dữ liệu đánh giá của người dùng
    user_ratings = df_ratings[df_ratings['User_id'] == user_id]
    if user_ratings.empty:
        return {"error": "Người dùng không có đánh giá."}
    
    # Thêm trọng số vào dữ liệu của người dùng
    user_data = user_ratings.merge(df_anime[['Anime_id', 'Weighted_JapaneseLevel']], on="Anime_id", how="left")
    user_data['Weighted_JapaneseLevel'] = user_data['Weighted_JapaneseLevel'].fillna(0)
    
    # Dự đoán
    predictions = decision_tree_model.predict(user_data[['User_id', 'Rating', 'Weighted_JapaneseLevel']])
    
    # Loại bỏ kết quả trùng lặp
    recommended_ids = list(dict.fromkeys(predictions))  # Anime_id đã được mã hóa (Name_encoded)
    unique_anime_ids = label_encoder.inverse_transform(recommended_ids)  # Giải mã ra Anime_id gốc
    
    # Lấy toàn bộ thông tin của anime
    recommended_anime = df_anime[df_anime['Anime_id'].isin(unique_anime_ids)]
    
    # Giới hạn số lượng kết quả trả về
    recommended_anime = recommended_anime.head(n)
    
    return recommended_anime.to_dict(orient="records")  # Trả về dưới dạng danh sách dict


# Endpoint cho /recommend2
@app.post("/")
async def deciontree(request: Request):
    data = await request.json()
    user_id = data.get("User_id")
    n = data.get("n", 10)  # Số lượng gợi ý mặc định là 10

    if user_id is None:
        return {"error": "Vui lòng cung cấp user_id"}

    # Gọi hàm recommend_by_decision_tree
    result = recommend_by_decision_tree(user_id, n)
    return {"recommendations": result}


import uvicorn
import os
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Render sẽ cung cấp cổng trong biến PORT
    uvicorn.run("decisiontree:app", host="0.0.0.0", port=port)

