"""Visible, logged preparation -> real-data check -> resumable full training."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile

BASE=Path(__file__).resolve().parents[1]
DATA=BASE.parents[1]/'dataset/coco-freeview'
OUTPUT=BASE/'models/hat-freeview-k6-seed42'


def download(url,path,archive=False,hash_prefix=None):
    if path.is_file():
        valid = zipfile.is_zipfile(path) if archive else True
        if hash_prefix:
            with path.open('rb') as stream:valid=hashlib.file_digest(stream,'sha256').hexdigest().startswith(hash_prefix)
        if valid:
            print(f'Уже загружено: {path.name}',flush=True);return
    path.parent.mkdir(parents=True,exist_ok=True)
    with (OUTPUT/'downloads.log').open('a') as errors:
        command=['curl','-L','--fail','--retry','5','--retry-delay','5','--connect-timeout','30','--max-time','86400','-C','-',url,'-o',str(path)]
        proc=subprocess.Popen(command,stdout=errors,stderr=errors)
        try:
            while proc.poll() is None:
                size=path.stat().st_size if path.exists() else 0
                print(f'Загрузка {path.name}: {size/1024**2:.1f} МБ | подробности downloads.log',flush=True)
                time.sleep(10)
        except KeyboardInterrupt:
            proc.terminate();proc.wait();raise
        if proc.returncode:raise RuntimeError(f'Загрузка не завершена: {path.name}; код {proc.returncode}. Повторный запуск продолжит загрузку.')
    if archive and not zipfile.is_zipfile(path):raise RuntimeError(f'Неполный архив {path.name}')


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    print('HAT — подготовка и обучение на реальных фиксациях COCO-FreeView',flush=True)
    print('K=6 | пакет=4 | максимум 5 эпох | DeepGaze сохранён и не участвует в обучении.',flush=True)
    print(f'Каталог результатов: {OUTPUT}',flush=True)
    from training.process_lock import process_lock
    with process_lock(OUTPUT/'launcher.lock'):
        for part in ('TP','TA'):
            name=f'COCOSearch18-images-{part}.zip'
            download(f'http://vision.cs.stonybrook.edu/~cvlab_download/{name}',DATA/'raw'/name,True)
        from torchvision.models import ResNet50_Weights
        torch_home=BASE/'models/torch';os.environ['TORCH_HOME']=str(torch_home)
        url=ResNet50_Weights.IMAGENET1K_V2.url
        weights=torch_home/'hub/checkpoints'/url.rsplit('/',1)[1]
        # torchvision itself verifies the expected hash during the first load.
        expected=weights.stem.rsplit('-',1)[1]
        download(url,weights,hash_prefix=expected)
        with weights.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if not digest.startswith(expected):raise RuntimeError('Контрольная сумма весов ImageNet не совпала; обучение не запускается.')
        from training.prepare_images import prepare
        prepare(DATA)
        args=[sys.executable,'-u','-m','training.train_hat','--data',str(DATA),'--output',str(OUTPUT),'--device','mps','--batch','4','--epochs','5']
        print('Проверяю реальный обучающий шаг и валидацию перед длинным запуском…',flush=True)
        subprocess.run(args+['--check-only'],cwd=BASE,check=True)
        print('Проверка завершена. Начинаю полное обучение.',flush=True)
        subprocess.run(args,cwd=BASE,check=True)

if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:
        print('\nПодготовка остановлена. Частичная загрузка сохранена. Запустите команду снова для продолжения.',flush=True)
        sys.exit(130)
    except Exception as exc:
        print(f'\nЗапуск остановлен: {exc}',flush=True)
        sys.exit(1)
