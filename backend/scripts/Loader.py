class CelebDFDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        # Iterate through Celeb-real and Celeb-fake
        for label, category in enumerate(['Celeb-real', 'Celeb-fake']):
            cat_path = os.path.join(root_dir, category)
            if not os.path.exists(cat_path): continue
            
            for vid_folder in os.listdir(cat_path):
                folder_path = os.path.join(cat_path, vid_folder)
                frames = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
                for frame_path in frames:
                    self.samples.append((frame_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)

# Preprocessing transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Initialize Dataset and Loaders
# Update path to match your D drive setup
DATA_PATH = r'D:\CHODER\Two_stream_Detection\backend\processed'
full_dataset = CelebDFDataset(root_dir=DATA_PATH, transform=transform)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

print(f"Dataset Loaded: {len(train_ds)} training images, {len(val_ds)} validation images.")