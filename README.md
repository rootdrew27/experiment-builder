# Experiment Builder

Description
---
Nothing here yet...

## Demo

```
import expb

DATA_PATH = r"path_to_data"
RANDOM_SEED = 477

expb.download_dataset(url="your_url", dest_path=DATA_PATH, is_zip=True) 

ds = expb.build_dataset(DATA_PATH, format="coco", task="segm")
```
```
# set the category hierarchy: control which category, when segmentations overlap, will be used in the label

# a background category is set to 0 by default
cat_hierarchy = {"grass": 1, "road": 2, "stop-sign": 3, "debris": 4, "misc": 4}

ds.metadata.set_category_hierarchy(cat_hierarchy)
```
```
# get subsets of the data using a ByOption (e.g. TAG, CATEGORY, IMGSHAPE)
import expb.By as By

good_ds = ds.subset(by=By.TAG, value="Good")

# split data into training and testing
train_ds, test_ds = good_ds.split([0.8, 0.2], shuffle=True, random_seed=RANDOM_SEED)
```
```
# Applying functions to a Dataset is easy. Heres an example of an rgb to grayscale function:
import numpy as np

def rgb2gray(data, weights=[0.299, 0.587, 0.114]):
    return np.dot(data[..., :3], weights)

train_ds.apply(rgb2gray)
# Pass parameters as a tuple or dict:

train_ds.apply(rgb2gray, params=([0.298, 0.591, 0.111],))
# OR
train_ds.apply(rgb2gray, kw_params={"weights": [0.298, 0.591, 0.111]})
```
```
# Importantly, data will not be loaded into memory and the function will not execute until .execute() is called:

tr_gray_ds = train_ds.execute(return_dataset=True) # if you'd like the metadata attached to your output, pass True to the execute function.

# Chaining actions and an execute call is permitted:

result = tr_gray_ds.apply(func1).apply(func2).execute(return_dataset=False)
```
```
# Performing operations on a gpu is easy with expb

# Use cupy's get_array_module to define functions impartial to the arrays location (e.g. host or gpu):

def rgb2gray(data, weights=[0.299, 0.587, 0.114]):
    xp = cp.get_array_module(data)
    if xp is cp:
        weights = cp.asarray(weights)
    return xp.dot(data[..., :3], weights)

# Then use .to() in your action chain

tr_gray_ds = train_ds.to('cuda').apply(rgb2gray).execute(return_dataset=True)
```

## Conceptual Structure (Core)

- **Experiment** The core element of this package, experiments are designed through, managed by, and run with this class.  
- **Dataset** Objects of this class are essentially beefy wrappers of NumPy arrays, with the added bonus of builtin memory mapping and batching. They are interfaces to the data and metadata of the dataset.
- **Apply/Execute Interface** Allows efficient and lazy operations to be performed on a Dataset.

## Progress

- [ ] Dataset Features
    - [x] Splitting Dataset
    - [x] Subsetting
    - [ ] Advanced Subsetting
    - [x] Apply/Execute Interface
    - [x] Download and Extract Dataset from zip (e.g. from Roboflow)
    - [x] Build Dataset from Annotation file (only Coco)
    - [x] Memory Mapping
    - [x] Use of cupy (i.e. GPU)
    - [x] Segmentation tasks
    - [x] Works with Torch (**Not Tested**)
    - [ ] Batching
    - [ ] Handle varying image shape.

- [ ] Experiment API
    - [ ] Logging
    - [ ] Profiling
    - [ ] Organization (directory and file creation)
    - [ ] Metrics

- [ ] Metadata
    - [x] Visualization
    - [x] Enable setting Category Hierarchies
    - [ ] Object Detection tasks
    - [ ] Build from other formats
    