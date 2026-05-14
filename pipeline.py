import torch
import numpy as np
import json
import joblib
import pandas as pd
import glob
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

def run_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("システムの準備をしています（モデル読み込み中...）")

    # ==========================================
    # 1. データの読み込みと準備
    # ==========================================
    CSV_DIR = "../data/track_data/train"
    HARMONIOUS_DIR = "../data/output_data/Harmonious_model"
    CLIP_FT_DIR = "../data/output_data/clip_model"

    # CSVから楽曲メタデータと、推論用の「候補ラベル（プレイリスト名）」を取得
    df_list = [pd.read_csv(f) for f in glob.glob(os.path.join(CSV_DIR, "*.csv"))]
    df = pd.concat(df_list, ignore_index=True)
    df['ジャンル'] = df['ジャンル'].fillna("unknown")
    track_db = df.drop_duplicates(subset=['id']).set_index('id')
    
    # eval.py のラベルとして、CSVに存在するプレイリスト名をすべて抽出
    scene_labels = df['プレイリスト名'].dropna().unique().tolist()

    # ==========================================
    # 2. AIモデルの読み込み
    # ==========================================
    # ① 画像→テキスト予測用（ファインチューニング済みモデル）
    model_ft = CLIPModel.from_pretrained(CLIP_FT_DIR).to(device)
    processor_ft = CLIPProcessor.from_pretrained(CLIP_FT_DIR)
    model_ft.eval()

    # ② テキスト→ID推論用（学習時に使ったベースモデル）
    # ※ベクトル空間を学習時と完全に一致させるため、K-Means用にはベースモデルを使用します
    base_model_name = "openai/clip-vit-base-patch32"
    model_base = CLIPModel.from_pretrained(base_model_name).to(device)
    processor_base = CLIPProcessor.from_pretrained(base_model_name)
    model_base.eval()

    # ③ K-Meansモデルと辞書の読み込み
    kmeans_models = joblib.load(os.path.join(HARMONIOUS_DIR, "kmeans_models.pkl"))
    with open(os.path.join(HARMONIOUS_DIR, "semantic_id_to_tracks_residual.json"), 'r', encoding='utf-8') as f:
        id_to_tracks = json.load(f)

    print("✅ 準備完了！\n")

    # ==========================================
    # 3. パイプライン各機能の定義
    # ==========================================
    def image_to_scene(image_path):
        """【Step 1】画像から最も適したプレイリスト名（シーン）を予測"""
        image = Image.open(image_path).convert("RGB")
        
        inputs = processor_ft(
            text=scene_labels, 
            images=image, 
            return_tensors="pt", 
            padding=True,
            truncation=True
        ).to(device)

        with torch.no_grad():
            outputs = model_ft(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)
            pred_idx = probs.argmax().item()
            
        predicted_scene = scene_labels[pred_idx]
        return predicted_scene

    def get_text_embeddings(text):
        """テキストをベースモデルでベクトル化"""
        inputs = processor_base(text=[text], return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            features = model_base.get_text_features(**inputs)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.cpu().numpy()[0]

    def scene_to_id(scene_text, target_genre):
        """【Step 2】シーンテキストとジャンルからSemantic IDを生成"""
        p_vec = get_text_embeddings(scene_text)
        
        if target_genre:
            t_vec = get_text_embeddings(target_genre)
        else:
            t_vec = np.zeros_like(p_vec)

        # 重み付けとfloat32変換（テストコードで成功した完璧な形）
        query_vector = np.concatenate([p_vec * 1.5, t_vec * 1.0])
        query_residual = query_vector.reshape(1, -1)
        
        predicted_tokens = []
        offset = 0

        for kmeans_model in kmeans_models:
            X_for_prediction = np.array(query_residual, dtype=np.float32)
            label = kmeans_model.predict(X_for_prediction)[0]
            token = f"<{label + offset}>"
            predicted_tokens.append(token)
            
            centroid = kmeans_model.cluster_centers_[label]
            query_residual = query_residual - centroid
            offset += kmeans_model.n_clusters

        return "".join(predicted_tokens)

    def recommend_music(predicted_id):
        """【Step 3】IDから楽曲を推薦"""
        recommended_tracks = id_to_tracks.get(predicted_id, [])
        
        if not recommended_tracks:
            print("⚠️ このベクトルに該当するIDの箱には曲が入っていませんでした。")
            return

        print(f"\n🎧 レコメンドされた楽曲 ({len(recommended_tracks)}曲):")
        for tid in recommended_tracks:
            if tid in track_db.index:
                t_info = track_db.loc[tid]
                print(f"  🎵 {t_info['曲名']} / {t_info['アーティスト']} (ジャンル: {t_info['ジャンル']})")

    # ==========================================
    # 4. メイン実行処理
    # ==========================================
    # ★ ここに入力したい画像のパスを指定してください
    test_image_path = "../data/image_data/test/sample/sample_img.jpg"  
    
    # ★ 絞り込みたいジャンルがあれば指定（なければ ""）
    test_genre = "j-pop"

    if not os.path.exists(test_image_path):
        print(f"エラー: 画像が見つかりません ({test_image_path})")
        return

    print("="*45)
    print("▶️ 画像から音楽をレコメンドするパイプライン")
    print("="*45)

    # 処理の実行
    predicted_scene = image_to_scene(test_image_path)
    print(f"🖼️ ① 画像分析結果 (Scene)  : {predicted_scene}")

    predicted_id = scene_to_id(predicted_scene, test_genre)
    print(f"🔢 ② 予測Semantic ID        : {predicted_id}")

    # ③ 楽曲の表示
    recommend_music(predicted_id)
    print("="*45)

if __name__ == "__main__":
    run_pipeline()