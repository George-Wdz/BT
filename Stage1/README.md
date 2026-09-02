# Stage1: Minute-Level Rainfall Retrieval

Stage1 estimates accumulated rainfall and rain probability for each 60-second window preceding a rain-gauge anchor. Inputs combine satellite-link measurements, link geometry, local temperature/humidity/pressure, and visual-weather probabilities.

## Quick Start

```bash
git clone https://github.com/George-Wdz/BT.git
cd BT
python -m pip install -r Stage1/requirements.txt

cd Stage1/minute_rain_retrieval
python check_delivery.py --training-only
python -m pytest -q
python train.py \
  --dataset-path data/reproducible_v1/minute_rainfall_full.npz \
  --output-dir outputs/reproduction \
  --epochs 80 \
  --batch-size 64 \
  --max-train-dry-ratio 3 \
  --selection-metric balanced_mae
```

The repository includes a fixed dataset and exported train/validation/test splits. Raw acquisition databases, camera images, and deployment weights are not required for offline training and evaluation.

See [README_CN.md](README_CN.md) for environment requirements, data construction, outputs, project structure, and limitations. Detailed model documentation is available in [minute_rain_retrieval/README_CN.md](minute_rain_retrieval/README_CN.md).
