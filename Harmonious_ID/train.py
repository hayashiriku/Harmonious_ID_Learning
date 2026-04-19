import torch
import numpy as np
import pandas as pd
import json
import joblib
import glob
import os
from sklearn.cluster import MiniBatchKMeans
from transformers import CLIPProcessor, CLIPModel
from collections import defaultdict

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ==========================================
    # 1. 複数CSVデータの読み込みと結合
    # ==========================================
    CSV_DIR = "../data/track_data/train"  # CSVファイルが入っているフォルダのパス
    
    csv_files = glob.glob(os.path.join(CSV_DIR, "*.csv"))
    if not csv_files:
        print(f"エラー: {CSV_DIR} フォルダにCSVファイルが見つかりません。")
        return

    print(f"{len(csv_files)} 件のCSVファイルを読み込みます...")
    df_list = []
    for file in csv_files:
        try:
            temp_df = pd.read_csv(file)
            df_list.append(temp_df)
            print(f"  - 読み込み成功: {os.path.basename(file)} ({len(temp_df)}曲)")
        except Exception as e:
            print(f"  - 読み込み失敗: {os.path.basename(file)} | エラー: {e}")

    # 重複削除を行わず、すべての「曲×プレイリスト」の組み合わせを維持する
    df = pd.concat(df_list, ignore_index=True).reset_index(drop=True)
    print(f"\nデータ結合完了: 全 {len(df)} 件の文脈を処理します")

    df['ジャンル'] = df['ジャンル'].fillna("unknown")
    df['track_context'] = df['ジャンル'] + " (" + df['アーティスト'] + ")"
    track_ids = df['id'].tolist()

    # ==========================================
    # 2. ベクトル生成 (CLIP)
    # ==========================================
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    def get_text_embeddings(text_list):
        inputs = processor(text=text_list, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model.get_text_features(**inputs)
            # 【追加】出力された「箱（オブジェクト）」の中から、テキストのベクトルを取り出す
            features = outputs.pooler_output
            # 取り出したベクトル（テンソル）に対して正規化を行う
            features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.cpu().numpy()

    print("\nベクトルの生成を開始します...")
    unique_playlists = df['プレイリスト名'].unique().tolist()
    
    playlist_embeddings = get_text_embeddings(unique_playlists)
    playlist_vec_map = {name: vec for name, vec in zip(unique_playlists, playlist_embeddings)}
    track_embeddings = get_text_embeddings(df['track_context'].tolist())

    hybrid_vectors = []
    for i, row in df.iterrows():
        p_vec = playlist_vec_map[row['プレイリスト名']]
        t_vec = track_embeddings[i]
        hybrid_vectors.append(np.concatenate([p_vec, t_vec]))
    
    hybrid_vectors = np.array(hybrid_vectors)

    # ==========================================
    # 3. 残差量子化 (Residual Quantization)
    # ==========================================
    print("\n残差量子化によるID生成を開始します...")
    N_CLUSTERS = [8, 4, 2] 
    
    residuals = hybrid_vectors.copy()
    num_layers = len(N_CLUSTERS)
    num_samples = len(hybrid_vectors)
    
    token_sequences = [[] for _ in range(num_samples)]
    all_special_tokens = set()
    models = []
    offset = 0

    for layer in range(num_layers):
        n_clusters = min(N_CLUSTERS[layer], num_samples)
        
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters, 
            random_state=42, 
            batch_size=max(10, min(1024, num_samples)),
            n_init="auto"
        )
        kmeans.fit(residuals)
        labels = kmeans.predict(residuals)
        centroids = kmeans.cluster_centers_
        
        models.append(kmeans)
        residuals = residuals - centroids[labels]
        
        for i, label in enumerate(labels):
            global_index = label + offset
            token = f"<{global_index}>"
            token_sequences[i].append(token)
            all_special_tokens.add(token)
            
        offset += n_clusters

    print("\n=== 保存処理 ===")
    
    # ★ ここで保存先のフォルダ名（パス）を指定します
    OUTPUT_DIR = "../data/output_data/Harmonious_model" 
    
    # フォルダが存在しない場合は自動で作成する
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    final_ids = defaultdict(list)
    id_to_tracks = defaultdict(list)

    for i, tid in enumerate(track_ids):
        id_str = "".join(token_sequences[i])
        
        if id_str not in final_ids[tid]:
            final_ids[tid].append(id_str)
            
        if tid not in id_to_tracks[id_str]:
            id_to_tracks[id_str].append(tid)

    # 保存先のフルパスを作成
    output_ids_path = os.path.join(OUTPUT_DIR, "track_ids_residual.json")
    output_to_tracks_path = os.path.join(OUTPUT_DIR, "semantic_id_to_tracks_residual.json")
    output_tokens_path = os.path.join(OUTPUT_DIR, "special_tokens_residual.json")
    output_model_path = os.path.join(OUTPUT_DIR, "kmeans_models.pkl")

    # 指定したフォルダ内に各種ファイルを保存
    with open(output_ids_path, 'w', encoding='utf-8') as f:
        json.dump(final_ids, f, ensure_ascii=False, indent=4)
        
    with open(output_to_tracks_path, 'w', encoding='utf-8') as f:
        json.dump(id_to_tracks, f, ensure_ascii=False, indent=4)

    sorted_tokens = sorted(list(all_special_tokens), key=lambda x: int(x.strip("<>")))
    with open(output_tokens_path, 'w', encoding='utf-8') as f:
        json.dump(sorted_tokens, f, ensure_ascii=False, indent=4)

    joblib.dump(models, output_model_path)

    print(f"以下のフォルダにすべてのファイルを保存しました: {OUTPUT_DIR}")

if __name__ == "__main__":
    train()