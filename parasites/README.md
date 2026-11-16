---
dataset_info:
  features:
  - name: image_id
    dtype: int64
  - name: image
    dtype: image
  - name: width
    dtype: int32
  - name: height
    dtype: int32
  - name: objects
    sequence:
    - name: id
      dtype: int64
    - name: area
      dtype: int64
    - name: bbox
      sequence: float32
      length: 4
    - name: category
      dtype:
        class_label:
          names:
            '0': parasites
            '1': Ancylostoma Spp
            '2': Ascaris Lumbricoides
            '3': Enterobius Vermicularis
            '4': Fasciola Hepatica
            '5': Hymenolepis
            '6': Schistosoma
            '7': Taenia Sp
            '8': Trichuris Trichiura
annotations_creators:
- crowdsourced
language_creators:
- found
language:
- en
license:
- cc
multilinguality:
- monolingual
size_categories:
- 1K<n<10K
source_datasets:
- original
task_categories:
- object-detection
task_ids: []
pretty_name: parasites-1s07h
tags:
- rf100
---

# Dataset Card for parasites-1s07h

** The original COCO dataset is stored at `dataset.tar.gz`**

## Dataset Description

- **Homepage:** https://universe.roboflow.com/object-detection/parasites-1s07h
- **Point of Contact:** francesco.zuppichini@gmail.com

### Dataset Summary

parasites-1s07h

### Supported Tasks and Leaderboards

- `object-detection`: The dataset can be used to train a model for Object Detection.

### Languages

English

## Dataset Structure

### Data Instances

A data point comprises an image and its object annotations.

```
{
  'image_id': 15,
  'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=640x640 at 0x2373B065C18>,
  'width': 964043,
  'height': 640,
  'objects': {
    'id': [114, 115, 116, 117], 
    'area': [3796, 1596, 152768, 81002],
    'bbox': [
      [302.0, 109.0, 73.0, 52.0],
      [810.0, 100.0, 57.0, 28.0],
      [160.0, 31.0, 248.0, 616.0],
      [741.0, 68.0, 202.0, 401.0]
    ], 
    'category': [4, 4, 0, 0]
  }
}
```

### Data Fields

- `image`: the image id
- `image`: `PIL.Image.Image` object containing the image. Note that when accessing the image column: `dataset[0]["image"]` the image file is automatically decoded. Decoding of a large number of image files might take a significant amount of time. Thus it is important to first query the sample index before the `"image"` column, *i.e.* `dataset[0]["image"]` should **always** be preferred over `dataset["image"][0]`
- `width`: the image width
- `height`: the image height
- `objects`: a dictionary containing bounding box metadata for the objects present on the image
  - `id`: the annotation id
  - `area`: the area of the bounding box
  - `bbox`: the object's bounding box (in the [coco](https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/#coco) format)
  - `category`: the object's category.


#### Who are the annotators?

Annotators are Roboflow users

## Additional Information

### Licensing Information

See original homepage https://universe.roboflow.com/object-detection/parasites-1s07h

### Citation Information

```
@misc{ parasites-1s07h,
    title = { parasites 1s07h Dataset },
    type = { Open Source Dataset },
    author = { Roboflow 100 },
    howpublished = { \url{ https://universe.roboflow.com/object-detection/parasites-1s07h } },
    url = { https://universe.roboflow.com/object-detection/parasites-1s07h },
    journal = { Roboflow Universe },
    publisher = { Roboflow },
    year = { 2022 },
    month = { nov },
    note = { visited on 2023-03-29 },
}"
```

### Contributions

Thanks to [@mariosasko](https://github.com/mariosasko) for adding this dataset.