import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os

def custom_collate(batch):
    # batchは [(image1, text1), (image2, text2), ...] のようなリスト
    # これを [image1, image2, ...], [text1, text2, ...] に分ける
    images = [item[0] for item in batch]
    texts = [item[1] for item in batch]
    return images, texts

# データの読み込み
class FolderBasedDriveDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = []

        # フォルダ内をスキャンして (画像パス, フォルダ名) のリストを作る
        # root_dir直下の各フォルダがプレイリスト名になる
        for folder_name in os.listdir(root_dir):
            if folder_name.startswith('.'):
                continue  # 隠しファイルやフォルダをスキップ
            folder_path = os.path.join(root_dir, folder_name)
            
            if os.path.isdir(folder_path):
                for img_name in os.listdir(folder_path):
                    # 画像ファイルのみを対象にする
                    if img_name.lower().endswith(('.png')):
                        img_path = os.path.join(folder_path, img_name)
                        self.samples.append((img_path, folder_name))
        
        print(f"読み込み完了: {len(self.samples)} 枚の画像, {len(os.listdir(root_dir))} つのプレイリスト")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, playlist_name = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"エラー: {img_path} をスキップします")
            image = Image.new("RGB", (224, 224), (0, 0, 0))
            
        return image, playlist_name
    
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 学習用データのパス
    train_dir = "../data/image_data/train"
    model_name = "openai/clip-vit-base-patch32"
    
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    
    dataset = FolderBasedDriveDataset(train_dir)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(5):
        total_loss = 0
        for images, texts in dataloader:
            # texts はフォルダ名のリスト（プレイリスト名）
            inputs = processor(
                text=list(texts), 
                images=images, 
                return_tensors="pt", 
                padding=True,
                truncation=True
            ).to(device)

            outputs = model(**inputs)
            
            logits_per_image = outputs.logits_per_image
            logits_per_text = outputs.logits_per_text
            labels = torch.arange(len(texts)).to(device)

            loss = (loss_img(logits_per_image, labels) + loss_txt(logits_per_text, labels)) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    # モデルの保存
    save_path = "../data/output_data/clip_model"
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    print(f"モデルを {save_path} に保存しました。")

if __name__ == "__main__":
    train()