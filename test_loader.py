from src.data import get_loaders

loaders, df = get_loaders("tb")

print("TB train batches:", len(loaders["train"]))
print("Images:", len(df))