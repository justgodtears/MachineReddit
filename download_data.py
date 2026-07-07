import kagglehub

# Download latest version
path = kagglehub.dataset_download("kaggle/reddit-comments-may-2015")

print("Path to dataset files:", path)