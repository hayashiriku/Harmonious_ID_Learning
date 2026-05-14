import numpy as np
import json
import joblib
import pandas as pd
import os
from sentence_transformers import SentenceTransformer

# ==========================================
# 定数定義
# ==========================================

OUTPUT_DIR = "../data/output_data/Japanese_Harmonious_model"  # train2.py でモデルやJSONを保存したフォルダのパス
CSV_PATH   = "../data/track_data/train/Playlists_filled_Genres_with_Lastfm_Tags.csv"  # 楽曲のメタデータが入ったCSVファイルのパス


def get_embeddings(model: SentenceTransformer, text_list: list, batch_size: int = 64) -> np.ndarray:
    """
    train.py と同じエンコード設定。passage: プレフィックスを付与する。
    """
    prefixed = [f"passage: {t}" for t in text_list]
    embeddings = model.encode(
        prefixed,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def predict_semantic_id(
    model: SentenceTransformer,
    kmeans_models: list,
    scene_text: str,
    genre: str,
) -> tuple[str, list]:
    """
    シーンテキストとジャンルから Semantic ID を予測する。

    train.py と同じベクトル構成:
      p_vec: シーンテキスト（プレイリスト名に相当）× 1.3
      t_vec: ジャンルのみ（推論時は曲名・アーティスト不明）× 1.0
    """
    # シーンベクトル
    p_vec = get_embeddings(model, [scene_text])[0]

    # ジャンルベクトル（未指定はゼロ埋め）
    if genre.strip():
        t_vec = get_embeddings(model, [f"genre: {genre}"])[0]
    else:
        t_vec = np.zeros(p_vec.shape, dtype=np.float32)

    # ハイブリッドベクトル（train.py と同じ重み）
    query_vector   = np.concatenate([p_vec * 1.3, t_vec * 1.0])
    query_residual = query_vector.reshape(1, -1).astype(np.float32)

    # 残差量子化で ID を予測
    predicted_tokens = []
    offset = 0
    for kmeans in kmeans_models:
        label = kmeans.predict(query_residual)[0]
        predicted_tokens.append(f"<{label + offset}>")
        query_residual = query_residual - kmeans.cluster_centers_[label]
        offset += kmeans.n_clusters

    return "".join(predicted_tokens), predicted_tokens


def verify():
    # ==========================================
    # 1. 必要なファイルの読み込み
    # ==========================================
    model_path  = os.path.join(OUTPUT_DIR, "kmeans_models.pkl")
    dict_path   = os.path.join(OUTPUT_DIR, "semantic_id_to_tracks_residual.json")

    for path in [model_path, dict_path]:
        if not os.path.exists(path):
            print(f"エラー: ファイルが見つかりません → {path}")
            return

    print("モデルとデータを読み込み中...")
    kmeans_models = joblib.load(model_path)
    with open(dict_path, 'r', encoding='utf-8') as f:
        id_to_tracks = json.load(f)

    # 楽曲メタデータ（結果表示用）
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        track_db = df.drop_duplicates(subset=['id']).set_index('id')
    else:
        print(f"⚠️  CSV ({CSV_PATH}) が見つかりません。曲名等の表示はスキップします。")
        track_db = pd.DataFrame()

    print("intfloat/multilingual-e5-large を読み込み中...")
    model = SentenceTransformer("intfloat/multilingual-e5-large")

    # ==========================================
    # 2. テスト入力
    #    ★ ここを変えて様々なシーンを検証できる
    #
    #    景観: 山道・森林 / 海沿い / 田舎 / 都市・市街地
    #    天気: 晴れ / 雨 / 曇り / 雪
    #    時間: 朝 / 昼 / 夕 / 夜
    #    季節: 春 / 夏 / 秋 / 冬
    # ==========================================
    scene_text = "春の昼の晴れた日の都市・市街地"  # シーンを自由に記述
    genre      = "j-pop"                           # ジャンル（不要なら "" を指定）

    print("\n" + "=" * 50)
    print("🔍 ドライブシーン楽曲推薦テスト")
    print("=" * 50)
    print(f"シーン  : {scene_text}")
    print(f"ジャンル: {genre or '指定なし'}")

    # ==========================================
    # 3. Semantic ID の予測
    # ==========================================
    predicted_id, tokens = predict_semantic_id(model, kmeans_models, scene_text, genre)

    layer_names = ["景観", "時間", "天気", "季節"]
    print(f"\n✅ 予測された Semantic ID: {predicted_id}")
    print("   内訳:")
    for name, token in zip(layer_names, tokens):
        print(f"     {name}: {token}")

    # ==========================================
    # 4. 楽曲推薦
    # ==========================================
    recommended_ids = id_to_tracks.get(predicted_id, [])

    if not recommended_ids:
        print("\n⚠️  このIDに紐づく楽曲がありません。")
        print("   ヒント: シーンテキストを学習データに近い表現に変えてみてください。")
        print(f"\n【登録済みID一覧（先頭10件）】")
        for sid in list(id_to_tracks.keys())[:10]:
            print(f"  {sid} → {len(id_to_tracks[sid])}曲")
        return

    print(f"\n🎧 推薦楽曲 ({len(recommended_ids)}曲):")
    print("-" * 50)

    for tid in recommended_ids:
        if track_db.empty or tid not in track_db.index:
            print(f"  track_id: {tid}")
            continue

        t_info = track_db.loc[tid]
        print(f"  🎵 {t_info['曲名']} / {t_info['アーティスト']}")

        # ジャンルがあれば表示
        if 'ジャンル' in t_info and pd.notna(t_info.get('ジャンル')):
            print(f"      ジャンル        : {t_info['ジャンル']}")

        # 類似アーティストがあれば表示
        if 'Similar_Artists' in t_info and pd.notna(t_info.get('Similar_Artists')):
            print(f"      類似アーティスト: {t_info['Similar_Artists']}")

        # どのプレイリストに属しているか
        playlists = df[df['id'] == tid]['プレイリスト名'].unique().tolist() if not track_db.empty else []
        if playlists:
            print(f"      収録プレイリスト: {' / '.join(playlists[:3])}")

    # ==========================================
    # 5. サマリー
    # ==========================================
    print("\n" + "=" * 50)
    print("📊 推薦サマリー")
    print("=" * 50)
    print(f"  シーン        : {scene_text}")
    print(f"  ジャンル      : {genre or '指定なし'}")
    print(f"  Semantic ID   : {predicted_id}")
    print(f"  推薦曲数      : {len(recommended_ids)}曲")


if __name__ == "__main__":
    verify()