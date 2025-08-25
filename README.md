# EVA-02 Segmentation

Минимальный пайплайн бинарной семантической сегментации на **ViT EVA-02** (через `timm`) с простым upsampling-декодером
и **AMP** (`torch.amp`) — без mmengine/mmseg.

**Возможности**

- Бэкбон: `eva02_*_patch14_448` из `timm` (Large/Base и др.)
- Простая голова: серия `Upsample → Conv → BN → ReLU` + `Conv1×1`
- Бинарный лосс: `BCEWithLogits + Dice`
- Метрики: **Dice score** (`1 − dice_loss`) и **IoU@0.5**
- Mixed precision: `torch.amp.autocast("cuda")` + `GradScaler`
- Аугментации: `albumentations` (Resize/Pad, Flip, Affine, Brightness/Contrast)

---

## Запуск

Базовый пример:

```bash
python train.py \
  --data_root data/flood \
  --backbone eva02_large_patch14_448 \
  --image_size 448 \
  --epochs 20 \
  --batch_size 4 \
  --lr 5e-5 \
  --num_workers 8 \
  --outdir runs_eva02_flood
```

## Что появится в `--outdir`

- `last.pt` — последний чекпоинт
- `best.pt` — лучший по **IoU**
- `val0_image.jpg`, `val0_pred.png`, `val0_mask.png` — визуализация

> **Примечание.** Входные изображения и маски ожидаются в каталоге  
> `data_root/{train,val}/{images,masks}/`.  
> Параметр `image_size` **должен быть кратен 14** (патч 14×14 у EVA-02).

---

## Аргументы CLI

- **`--data_root`** *(str, по умолч. `data/flood`)*  
  Путь к корню датасета. Структура:

```
train/images/*.jpg
train/masks/*.png
val/images/*.jpg
val/masks/*.png
```

- **`--backbone`** *(str, по умолч. `eva02_large_patch14_448`)*  
  Имя модели из `timm`. Примеры: `eva02_large_patch14_448`, `eva02_base_patch14_448`.  
  **Large** — точнее, **Base** — быстрее/легче.

- **`--image_size`** *(int, по умолч. `448`)*  
  Сторона входного изображения после аугментаций. **Кратно 14** (например: `448`, `392`, `336`, `294`).  
  Меньше размер → быстрее и экономнее по памяти.

- **`--epochs`** *(int, по умолч. `20`)*  
  Количество эпох обучения.

- **`--batch_size`** *(int, по умолч. `4`)*  
  Размер батча. Увеличивайте при наличии памяти; уменьшайте при OOM.

- **`--lr`** *(float, по умолч. `5e-5`)*  
  Learning rate для `AdamW`.

- **`--bce_weight`** *(float ∈ [0,1], по умолч. `0.5`)*  
  Вес в комбинированном лоссе:
  ```loss = bce_weight * BCEWithLogits + (1 - bce_weight) * Dice```
  Ближе к 1 — больше вклад BCE, ближе к 0 — больше вклад Dice.

- **`--seed`** *(int, по умолч. `42`)*  
  Фиксация сидов для воспроизводимости.

- **`--num_workers`** *(int, по умолч. `8`)*  
  Количество воркеров `DataLoader`.  
  На Windows иногда быстрее `0`; на Linux — `4–16` (зависит от CPU/диска).

- **`--outdir`** *(str, по умолч. `runs_eva02_flood`)*  
  Папка для сохранения чекпоинтов и примеров инференса.
