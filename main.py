from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import pandas as pd
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

workers = 0 if os.name == 'nt' else 4
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

from PIL import Image
def collate_fn(x):
    return x[0]

dataset = datasets.ImageFolder('./images')
print({i:c for c, i in dataset.class_to_idx.items()})
dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

aligned = []
names = []
for x, y in loader:
    x_aligned, prob = mtcnn(x, return_prob=True)
    if x_aligned is not None:
        print('Face detected with probability: {:8f}'.format(prob))
        aligned.append(x_aligned)

        names.append(dataset.idx_to_class[y])

aligned = torch.stack(aligned).to(device)
embeddings = resnet(aligned)

dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
print(pd.DataFrame(dists, columns=names, index=names))
img = Image.open("images/Dwight/Dwight2.jpg")


img_cropped = mtcnn(img, save_path="images/Michael/6765d052-e971-45c6-9763-c069b4f523b7.jpg",)
aligned = torch.stack([img_cropped]).to(device)

img_embedding = resnet(aligned)
for embedding, name in zip(embeddings,names):
    print((embedding-img_embedding[0]).norm().item(),name)