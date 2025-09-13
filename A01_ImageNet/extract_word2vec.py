import fasttext
import fasttext.util
import os
from pathlib import Path
from Deps.CustomFuctions.classes import IMAGENET2012_CLASSES
from collections import OrderedDict
import torch
from tqdm import tqdm

script_dir = Path(__file__).parent
project_root = Path(__file__).resolve().parent.parent

def extract_before_comma(input_string):
    comma_index = input_string.find(',')
    if comma_index != -1:
        return input_string[:comma_index]
    else:
        return input_string

ft = fasttext.load_model(os.path.join(project_root, "Deps", "corpora", "cc.en.300.bin"))
# fasttext.util.download_model('en', if_exists='ignore', datapath=os.path.join(project_root, "Deps", "corpora"))
fasttext.util.reduce_model(ft, 20)

FastTextFirstEmbeddings = OrderedDict()

for key, value in tqdm(IMAGENET2012_CLASSES.items()):
    FastTextFirstEmbeddings[key] = torch.tensor(ft.get_word_vector(extract_before_comma(value)))

torch.save(FastTextFirstEmbeddings, os.path.join(project_root, "Results", "word2vec", 'ImageNet1k_FirstEmbeddings_Dict.pt'))

# Generate tensor format for direct use in init_symbol_set
embedding_list = []
for key, value in IMAGENET2012_CLASSES.items():
    embedding_vector = torch.tensor(ft.get_word_vector(extract_before_comma(value)))
    embedding_list.append(embedding_vector)

# Stack all embeddings into a 1000x20 tensor
ImageNet1k_FirstEmbeddings_Tensor = torch.stack(embedding_list, dim=0)
print(f"Generated tensor shape: {ImageNet1k_FirstEmbeddings_Tensor.shape}")

# Save the tensor format
torch.save(ImageNet1k_FirstEmbeddings_Tensor, os.path.join(project_root, "Results", "word2vec", 'ImageNet1k_FirstEmbeddings_Tensor.pt'))



