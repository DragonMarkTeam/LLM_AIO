from load_dataset import load_data

path="tatsu-lab/alpaca"
split="train"
cache_dir="/content/drive/MyDrive/AIO/VLSP2023/dataset/alpaca"
save_path="/content/drive/MyDrive/AIO/VLSP2023/dataset/save/alpaca"
load_data(path, save_path, cache_dir, split)