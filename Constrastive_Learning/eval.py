import torch
from PIL import Image
import os
from transformers import CLIPProcessor, CLIPModel

def evaluate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # train.py で保存したモデルのパス
    model_path = "../data/output_data/clip_model"  
    # テスト用画像が入っているフォルダのパス
    test_dir = "../data/image_data/test"        
    
    # 保存したモデルとプロセッサをロード
    model = CLIPModel.from_pretrained(model_path).to(device)
    processor = CLIPProcessor.from_pretrained(model_path)
    
    # テストフォルダ内の全ラベル（フォルダ名）を取得（隠しファイルを除外）
    labels = []
    for f in os.listdir(test_dir):
        if not f.startswith('.') and os.path.isdir(os.path.join(test_dir, f)):
            labels.append(f)
            
    if len(labels) == 0:
        print("エラー: テスト用のプレイリスト（フォルダ）が見つかりません。")
        return
    
    model.eval()
    correct, total = 0, 0

    print("--- 検証開始 ---")

    with torch.no_grad():
        for label_name in labels:
            folder_path = os.path.join(test_dir, label_name)
            
            for img_name in os.listdir(folder_path):
                if img_name.startswith('.'): continue
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')): continue
                
                # 画像のフルパスを取得
                img_path = os.path.join(folder_path, img_name)
                
                # 画像の読み込み
                image = Image.open(img_path).convert("RGB")
                
                inputs = processor(
                    text=labels, 
                    images=image, 
                    return_tensors="pt", 
                    padding=True,
                    truncation=True
                ).to(device)

                outputs = model(**inputs)

                probs = outputs.logits_per_image.softmax(dim=1)
                pred_idx = probs.argmax().item()
                predicted_label = labels[pred_idx]
                
                # 正解判定
                if predicted_label == label_name:
                    correct += 1
                else:
                    # ★追加：間違えた場合の詳細情報を出力
                    print(f"【誤判定】")
                    print(f"  画像パス: {img_path}")
                    print(f"  正解のフォルダ: {label_name}")
                    print(f"  AIの予測結果  : {predicted_label}")
                    print("-" * 30)
                    
                total += 1

    if total > 0:
        print(f"\n検証完了！ 正解数: {correct}/{total} | 精度: {correct/total:.2%}")
    else:
        print("テスト可能な画像が見つかりませんでした。")

if __name__ == "__main__":
    evaluate()