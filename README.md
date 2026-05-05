# Car Damage Detection and Cost Estimation (Flask + CNN)

This project is a Flask app for vehicle damage analysis with a 3-stage model pipeline:

1. Car presence detection (`car` / `no_car`)
2. Damage detection (`damaged` / `undamaged`)
3. Part classification (`bonnet`, `bumper`, `door`, `fender`, `light`, `windshield`)

The app now supports:
- multi-image upload
- live webcam detection
- uncertainty-aware part prediction (Top-3 suggestions)
- CSV-based report storage

## Detection Order (Important)
The system **always checks car presence first**.  
If no car is detected, part/damage estimation is skipped for that frame/image.

## Project Structure
- `app.py`
- `utils/model_utils.py`
- `utils/csv_utils.py`
- `utils/training_utils.py`
- `train_car_presence.py`
- `train_damage.py`
- `train_part.py`
- `train_all_models.py`
- `prepare_dataset_splits.py`
- `templates/`
- `static/uploads/`
- `models/`
- `data/`

## Required Trained Models
After training, these files should exist in `models/`:
- `car_presence_model.keras`
- `damage_model.keras`
- `part_classifier_mobilenetv2.keras`

## Dataset Layout
Create datasets like this:

```text
datasets/
  car_presence/
    train/
      car/
      no_car/
    val/
      car/
      no_car/
  damage/
    train/
      damaged/
      undamaged/
    val/
      damaged/
      undamaged/
  parts/
    train/
      bonnet/
      bumper/
      door/
      fender/
      light/
      windshield/
    val/
      bonnet/
      bumper/
      door/
      fender/
      light/
      windshield/
```

## If You Have Raw Class Folders
If your raw data is in class folders (without train/val), create splits using:

```bash
python prepare_dataset_splits.py --source-dir raw_datasets/parts --output-dir datasets/parts --clear-output
python prepare_dataset_splits.py --source-dir raw_datasets/damage --output-dir datasets/damage --clear-output
python prepare_dataset_splits.py --source-dir raw_datasets/car_presence --output-dir datasets/car_presence --clear-output
```

## Training

### Train All Models in Correct Pipeline Order
```bash
python train_all_models.py --dataset-root datasets --models-dir models
```

### Train Individual Models
```bash
python train_car_presence.py --data-dir datasets/car_presence --model-out models/car_presence_model.keras
python train_damage.py --data-dir datasets/damage --model-out models/damage_model.keras
python train_part.py --data-dir datasets/parts --model-out models/part_classifier_mobilenetv2.keras
```

## Run App
```bash
python app.py
```
Open: `http://127.0.0.1:5000`

## Accuracy Notes
- Accurate part detection depends heavily on dataset quality and class balance.
- Use clear close-up images for each part class and include multiple angles/lighting.
- If model files are missing, app enters fallback mode and accuracy will drop.
