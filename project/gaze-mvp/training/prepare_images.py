"""Prepare only required official padded images, without unsafe zip extraction."""
import json
from pathlib import Path
import time
import zipfile
from PIL import Image


def prepare(root, stop_requested=lambda: False):
    root=Path(root);raw=root/'raw';out=root/'images';out.mkdir(exist_ok=True)
    names={r['name'] for r in json.loads((raw/'COCOFreeView_fixations_trainval.json').read_text())}
    for part in ('TP','TA'):
        archive=raw/f'COCOSearch18-images-{part}.zip'
        deadline=time.monotonic()+86400
        while True:
            if stop_requested():raise InterruptedError('Подготовка остановлена')
            try:
                with zipfile.ZipFile(archive) as z: members=z.infolist()
                break
            except (OSError,zipfile.BadZipFile):
                if time.monotonic()>deadline:raise RuntimeError(f'archive incomplete: {archive}')
                size=archive.stat().st_size/1024**2 if archive.exists() else 0
                print(f'Подготовка: ждём загрузку {part}, получено {size:.1f} МБ',flush=True);time.sleep(15)
        with zipfile.ZipFile(archive) as z:
            for member in members:
                name=Path(member.filename).name
                if name not in names:continue
                dest=out/name
                if dest.exists():continue
                data=z.read(member)
                import io
                with Image.open(io.BytesIO(data)) as im:
                    if im.size!=(1680,1050):raise ValueError(f'{name}: unexpected canvas {im.size}')
                    im.verify()
                temp=out/(name+'.partial');temp.write_bytes(data);temp.replace(dest)
        print(f'Подготовлена часть {part}',flush=True)
    missing=[n for n in names if not (out/n).exists()]
    if missing:raise ValueError(f'{len(missing)} missing images, first: {missing[:3]}')
    print(f'Проверены {len(names)} фотографий на полотне 1680×1050.',flush=True)
    return out
