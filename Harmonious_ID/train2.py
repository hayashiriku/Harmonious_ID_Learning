import numpy as np
import pandas as pd
import json
import joblib
import os
from sklearn.cluster import MiniBatchKMeans
from sentence_transformers import SentenceTransformer
from collections import defaultdict

# ==========================================
# 定数定義
# ==========================================

# 各要素のクラスタ数（論文のRQ層に対応）
# Layer 1: 景観（山道・森林 / 海沿い / 田舎 / 都市・市街地）← 山道と森林は1まとめ
# Layer 2: 天気（晴れ・雨・曇り・雪）
# Layer 3: 時間（朝・昼・夕・夜）
# Layer 4: 季節（春・夏・秋・冬）
N_CLUSTERS = [4, 4, 4, 4]  # 最大256通り

# 全データが入った単一CSVのパス
# 列構成: 検索キーワード, プレイリスト名, プレイリストID, 曲名, id, アーティスト, アーティストID
#         + ジャンル, Similar_Artists（追加済みの場合）
CSV_PATH = "../data/track_data/train/Playlists_filled_Genres_with_Lastfm_Tags.csv"


def build_scene_text(playlist_name: str) -> str:
    """
    プレイリスト名そのものをシーンテキストとして使用する。
    手動定義やキーワードマッピングは行わず、テキストの意味空間に任せる。

    プレイリスト名が持つ意味をmultilingual-e5がそのままベクトル化するため、
    「晴れた日のドライブ」は天気情報を、「夜のドライブ」は時間情報を
    自然に反映したベクトルになる。
    キーワードと異なるプレイリスト名も正しい意味空間に配置される。
    """
    return playlist_name


def build_track_context(row: pd.Series) -> str:
    """
    トラックのコンテキストテキストを構築する。
    曲名・アーティストを基本とし、ジャンル・類似アーティストが
    あれば追加する。欠損している場合は自動的にスキップする。

    推論時の入力（シーンテキスト＋ジャンル）との乖離を最小化するため、
    Similar_Tracks は含めない。
    """
    parts = []
    if pd.notna(row.get('アーティスト')):
        parts.append(f"artist: {row['アーティスト']}")
    if pd.notna(row.get('曲名')):
        parts.append(f"track: {row['曲名']}")
    if pd.notna(row.get('ジャンル')) and str(row['ジャンル']).strip():
        parts.append(f"genre: {row['ジャンル']}")
    if pd.notna(row.get('Similar_Artists')) and str(row['Similar_Artists']).strip():
        parts.append(f"similar artists: {row['Similar_Artists']}")
    return " | ".join(parts)


def get_embeddings(model: SentenceTransformer, text_list: list, batch_size: int = 64) -> np.ndarray:
    """
    multilingual-e5-large 用のエンコード。
    passage: プレフィックスを付与する。
    """
    prefixed = [f"passage: {t}" for t in text_list]
    embeddings = model.encode(
        prefixed,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def train():
    # ==========================================
    # 1. 単一CSVの読み込み
    # ==========================================
    if not os.path.exists(CSV_PATH):
        print(f"エラー: CSVファイルが見つかりません → {CSV_PATH}")
        return

    print(f"CSVファイルを読み込み中: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH).reset_index(drop=True)
    print(f"読み込み完了: {len(df)} 件")

    # 検索キーワード別の件数を表示
    print("\n【検索キーワード別 件数】")
    for kw, count in df['検索キーワード'].value_counts().items():
        print(f"  {kw}: {count}曲")

    # ジャンル・類似アーティストの充足率を表示
    for col in ['ジャンル', 'Similar_Artists']:
        if col in df.columns:
            rate = df[col].notna().mean() * 100
            print(f"  {col} 充足率: {rate:.1f}%")
        else:
            df[col] = np.nan

    # ==========================================
    # 2. テキストコンテキストの構築
    # ==========================================
    print("\nテキストコンテキストを構築中...")

    # シーンテキスト: プレイリスト名のみ（意味空間に任せる）
    df['scene_text'] = df['プレイリスト名'].apply(build_scene_text)
    # トラックコンテキスト: 曲名 + アーティスト + ジャンル + 類似アーティスト
    df['track_context'] = df.apply(build_track_context, axis=1)

    print("\n【シーンテキストサンプル】")
    for ctx in df['scene_text'].head(3):
        print(f"  {ctx}")

    print("\n【トラックコンテキストサンプル】")
    for ctx in df['track_context'].head(3):
        print(f"  {ctx}")

    track_ids = df['id'].tolist()
    unique_scenes = df['scene_text'].unique().tolist()

    # ==========================================
    # 3. ベクトル生成 (multilingual-e5-large)
    # ==========================================
    print("\nモデルを読み込み中: intfloat/multilingual-e5-large")
    model = SentenceTransformer("intfloat/multilingual-e5-large")

    print("\nシーンベクトルを生成中...")
    scene_embeddings = get_embeddings(model, unique_scenes)
    scene_vec_map = {s: v for s, v in zip(unique_scenes, scene_embeddings)}

    print("\nトラックベクトルを生成中...")
    track_embeddings = get_embeddings(model, df['track_context'].tolist())

    # ==========================================
    # 4. ハイブリッドベクトルの構築
    #    シーン（プレイリスト名）を 1.5 倍に重み付け
    #    トラック（曲名+アーティスト+ジャンル+類似アーティスト）を 1.0 倍
    # ==========================================
    print("\nハイブリッドベクトルを構築中...")
    hybrid_vectors = np.array([
        np.concatenate([
            scene_vec_map[df.iloc[i]['scene_text']] * 1.3,
            track_embeddings[i] * 1.0,
        ]).astype(np.float32)
        for i in range(len(df))
    ])
    print(f"ハイブリッドベクトルのshape: {hybrid_vectors.shape}")

    # ==========================================
    # 5. 残差量子化 (Residual Quantization)
    #    Layer 1: 景観(4) → Layer 2: 天気(4) → Layer 3: 時間(4) → Layer 4: 季節(4)
    # ==========================================
    print(f"\n残差量子化によるSemantic ID生成を開始します...")
    print(f"  Layer構成: 景観({N_CLUSTERS[0]}) × 天気({N_CLUSTERS[1]}) × 時間({N_CLUSTERS[2]}) × 季節({N_CLUSTERS[3]})")
    print(f"  最大ID組み合わせ数: {N_CLUSTERS[0]*N_CLUSTERS[1]*N_CLUSTERS[2]*N_CLUSTERS[3]}")

    residuals = hybrid_vectors.copy()
    num_samples = len(hybrid_vectors)
    layer_names = ["景観", "時間", "天気", "季節"]

    token_sequences = [[] for _ in range(num_samples)]
    all_special_tokens = set()
    kmeans_models = []
    offset = 0

    for layer, (n_clusters_raw, layer_name) in enumerate(zip(N_CLUSTERS, layer_names)):
        n_clusters = min(n_clusters_raw, num_samples)
        print(f"  Layer {layer + 1}/{len(N_CLUSTERS)} ({layer_name}): {n_clusters} クラスタ")

        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=42,
            batch_size=max(10, min(1024, num_samples)),
            n_init="auto",
        )
        kmeans.fit(residuals)
        labels = kmeans.predict(residuals)
        centroids = kmeans.cluster_centers_

        kmeans_models.append(kmeans)
        residuals = residuals - centroids[labels]

        for i, label in enumerate(labels):
            token = f"<{label + offset}>"
            token_sequences[i].append(token)
            all_special_tokens.add(token)

        offset += n_clusters

    # ==========================================
    # 6. 保存処理
    # ==========================================
    OUTPUT_DIR = "../data/output_data/Japanese_Harmonious_model"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n=== 保存処理 → {OUTPUT_DIR} ===")

    final_ids: dict = defaultdict(list)
    id_to_tracks: dict = defaultdict(list)

    for i, tid in enumerate(track_ids):
        id_str = "".join(token_sequences[i])
        if id_str not in final_ids[tid]:
            final_ids[tid].append(id_str)
        if tid not in id_to_tracks[id_str]:
            id_to_tracks[id_str].append(tid)

    paths = {
        "track_ids":      os.path.join(OUTPUT_DIR, "track_ids_residual.json"),
        "id_to_tracks":   os.path.join(OUTPUT_DIR, "semantic_id_to_tracks_residual.json"),
        "special_tokens": os.path.join(OUTPUT_DIR, "special_tokens_residual.json"),
        "kmeans_models":  os.path.join(OUTPUT_DIR, "kmeans_models.pkl"),
        "scene_vec_map":  os.path.join(OUTPUT_DIR, "scene_vec_map.pkl"),
    }

    with open(paths["track_ids"], 'w', encoding='utf-8') as f:
        json.dump(dict(final_ids), f, ensure_ascii=False, indent=4)

    with open(paths["id_to_tracks"], 'w', encoding='utf-8') as f:
        json.dump(dict(id_to_tracks), f, ensure_ascii=False, indent=4)

    sorted_tokens = sorted(list(all_special_tokens), key=lambda x: int(x.strip("<>")))
    with open(paths["special_tokens"], 'w', encoding='utf-8') as f:
        json.dump(sorted_tokens, f, ensure_ascii=False, indent=4)

    joblib.dump(kmeans_models, paths["kmeans_models"])
    joblib.dump(scene_vec_map, paths["scene_vec_map"])

    # ==========================================
    # 7. 簡易レポート
    # ==========================================
    unique_ids = set("".join(seq) for seq in token_sequences)
    print(f"\n=== 生成結果サマリー ===")
    print(f"  処理曲数              : {num_samples}")
    print(f"  ユニークID数          : {len(unique_ids)}")
    print(f"  特殊トークン総数      : {len(all_special_tokens)}")
    print(f"  最大ID組み合わせ数    : {N_CLUSTERS[0]*N_CLUSTERS[1]*N_CLUSTERS[2]*N_CLUSTERS[3]}")

    print(f"\n【IDサンプル（先頭5件）】")
    for i in range(min(5, num_samples)):
        row = df.iloc[i]
        id_str = "".join(token_sequences[i])
        print(f"  {row['曲名']} / {row['アーティスト']}")
        print(f"    シーン      : {row['scene_text']}")
        print(f"    コンテキスト: {row['track_context']}")
        print(f"    ID          : {id_str}")

    print(f"\n【検索キーワード別 ID分布】")
    for keyword in df['検索キーワード'].unique():
        mask = df['検索キーワード'] == keyword
        ids_in_kw = set("".join(token_sequences[i]) for i in df[mask].index)
        print(f"  {keyword}: {len(ids_in_kw)} ユニークID")

    print(f"\n✅ すべてのファイルを保存しました: {OUTPUT_DIR}")


if __name__ == "__main__":
    train()