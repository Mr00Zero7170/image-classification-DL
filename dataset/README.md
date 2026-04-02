# Dataset

## CIFAR-10

This project uses **CIFAR-10**, a benchmark dataset curated by the
[Canadian Institute for Advanced Research](https://www.cs.toronto.edu/~kriz/cifar.html).

| Property         | Value                                      |
|------------------|--------------------------------------------|
| Classes          | 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) |
| Training images  | 50,000                                     |
| Test images      | 10,000                                     |
| Image size       | 32 × 32 pixels, RGB                        |
| File size        | ~163 MB                                    |

## Automatic Download

**No manual download required.** `train.py` and the notebook call
`tf.keras.datasets.cifar10.load_data()`, which automatically downloads and
caches the dataset to `~/.keras/datasets/` on first run.

```bash
python src/train.py --model cnn   # dataset is fetched automatically
```

## Manual Download (optional)

If you prefer to download manually:

1. Visit https://www.cs.toronto.edu/~kriz/cifar.html
2. Download **CIFAR-10 Python version** (cifar-10-python.tar.gz)
3. Extract into this `dataset/` folder:
   ```
   dataset/
     cifar-10-batches-py/
       data_batch_1 … data_batch_5
       test_batch
       batches.meta
   ```

## Using a Custom Dataset

To use your own image dataset instead of CIFAR-10:

1. Organise images into class subfolders:
   ```
   dataset/
     custom/
       train/
         cat/  img1.jpg  img2.jpg …
         dog/  img1.jpg  img2.jpg …
       test/
         cat/  …
         dog/  …
   ```
2. Use `tf.keras.preprocessing.image_dataset_from_directory()` in `utils.py`
   to load your dataset and replace the `load_cifar10()` call in `train.py`.
