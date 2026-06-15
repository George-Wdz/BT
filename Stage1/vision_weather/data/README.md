# Vision Weather Data

This directory is the default local data root for the Stage1 weather image classifier.

- Put labeled classification images under `raw/<class_name>/`, for example `raw/sunny`, `raw/cloudy`, and `raw/rain`.
- Use `../prepare_weather_split.py` to create `split/train`, `split/val`, and `split/test`.
- Large image files are intentionally ignored by Git. Keep source images and generated split images local or publish them as external artifacts.
