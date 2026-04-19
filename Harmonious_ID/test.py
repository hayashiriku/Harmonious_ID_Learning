import torch
import numpy as np
import json
import joblib
import pandas as pd
import glob
import os
from transformers import CLIPProcessor, CLIPModel

def verify_kmeans_inference():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ==========================================
    # 1. 必要なファイル・モデルの読み込み
    # ==========================================
    OUTPUT_DIR = "../data/output_data/Harmonious_model"  # train.py でモデルやJSONを保存したフォルダのパス
    CSV_DIR = "../data/track_data/train"  # 楽曲のメタデータが入ったCSVファイルがあるフォルダのパス
    
    print("モデルとデータを読み込んでいます...")
    # 学習時に使ったのと同じCLIPモデルを指定
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    # K-MeansモデルとID辞書の読み込み
    model_path = os.path.join(OUTPUT_DIR, "kmeans_models.pkl")
    dict_path = os.path.join(OUTPUT_DIR, "semantic_id_to_tracks_residual.json")
    
    if not os.path.exists(model_path) or not os.path.exists(dict_path):
        print("エラー: 必要なモデルやJSONファイルが見つかりません。")
        return

    kmeans_models = joblib.load(model_path)
    with open(dict_path, 'r', encoding='utf-8') as f:
        id_to_tracks = json.load(f)

    # 楽曲のメタデータ表示用
    df_list = [pd.read_csv(f) for f in glob.glob(os.path.join(CSV_DIR, "*.csv"))]
    df = pd.concat(df_list, ignore_index=True)
    df['ジャンル'] = df['ジャンル'].fillna("unknown")
    track_db = df.drop_duplicates(subset=['id']).set_index('id')

    def get_text_embeddings(text_list):
        inputs = processor(text=text_list, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model.get_text_features(**inputs)
            features = outputs.pooler_output
            features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.cpu().numpy()

    # ==========================================
    # 2. テスト入力（検証したいプレイリストとジャンル）
    # ==========================================
    # ★ ここに検証したいプレイリスト名を入力してください
    target_playlist = "Songs I want to listen to at sunrise"
    
    # ★ ここに、そのプレイリストの中から絞り込みたいジャンルや特徴を入力してください
    # （例: "j-pop", "anime", またはジャンル指定なしなら ""）
    target_genre = "j-pop"

    print("\n" + "="*45)
    print("🔍 K-Means推論テスト")
    print("="*45)
    print(f"入力プレイリスト : {target_playlist}")
    print(f"入力ジャンル     : {target_genre if target_genre else '指定なし'}")

    # ==========================================
    # 3. 検索用ハイブリッド・ベクトルの生成
    # ==========================================
    # ① プレイリスト名のベクトル化
    p_vec = get_text_embeddings([target_playlist])[0]

    # ② ジャンルのベクトル化（学習時の track_context に相当するもの）
    if target_genre:
        t_vec = get_text_embeddings([target_genre])[0]
    else:
        t_vec = np.zeros_like(p_vec) # 指定なしの場合はゼロ埋め

    # 結合してK-Meansに入力するクエリを作成
    query_vector = np.concatenate([p_vec, t_vec])
    query_residual = query_vector.reshape(1, -1)

    # ==========================================
    # 4. K-Meansモデルによる Semantic ID の予測
    # ==========================================
    predicted_tokens = []
    offset = 0

    for kmeans_model in kmeans_models:
        label = kmeans_model.predict(query_residual)[0]
        token = f"<{label + offset}>"
        predicted_tokens.append(token)
        
        centroid = kmeans_model.cluster_centers_[label]
        query_residual = query_residual - centroid
        offset += kmeans_model.n_clusters

    predicted_id = "".join(predicted_tokens)
    print(f"\n✅ K-Meansが予測したID: {predicted_id}")

    # ==========================================
    # 5. レコメンド結果の検証
    # ==========================================
    recommended_tracks = id_to_tracks.get(predicted_id, [])

    if not recommended_tracks:
        print("⚠️ このベクトルに該当するIDの箱には曲が入っていませんでした。")
    else:
        print(f"\n🎧 レコメンドされた楽曲 ({len(recommended_tracks)}曲):")
        
        correct_count = 0
        for tid in recommended_tracks:
            if tid in track_db.index:
                t_info = track_db.loc[tid]
                
                # その曲が実際にターゲットのプレイリストに含まれているか判定
                # ※CSV内に複数の同じ曲がある場合を考慮して元のdfから検索
                is_in_playlist = not df[(df['id'] == tid) & (df['プレイリスト名'] == target_playlist)].empty
                
                if is_in_playlist:
                    mark = "🟢 [正解]"
                    correct_count += 1
                else:
                    mark = "🔴 [別リスト]"
                    
                print(f"  {mark} {t_info['曲名']} / {t_info['アーティスト']}")
                print(f"      - ジャンル: {t_info['ジャンル']}")
        
        # 精度の表示
        accuracy = (correct_count / len(recommended_tracks)) * 100
        print("\n" + "-"*45)
        print(f"📊 検証結果: レコメンドされた曲のうち、指定したプレイリストの曲だった割合")
        print(f"   => {correct_count} / {len(recommended_tracks)} 曲 ({accuracy:.1f}%)")

if __name__ == "__main__":
    verify_kmeans_inference()