from datasets import load_dataset
from pathlib import Path

parquet_files = sorted(Path("image_ads").glob("*.parquet"))

print(f"Найдено parquet-файлов: {len(parquet_files)}")

if not parquet_files:
    raise RuntimeError("Parquet-файлы не найдены")

print("Первый файл:", parquet_files[0])

ds = load_dataset(
    "parquet",
    data_files=[str(p) for p in parquet_files],
    split="train"
)

out = Path("pitt_ads_images")
out.mkdir(exist_ok=True)

for i, item in enumerate(ds):
    img = item["image"].convert("RGB")
    img.save(out / f"{i:06d}.jpg", quality=95)

    if i % 1000 == 0:
        print(f"Сохранено: {i}")

print(f"Готово. Всего изображений: {len(ds)}")
