# Semantic Segmentation Template
This repository provides a template designed to help you start segmentation tasks immediately.   

The core principles of this repository are:  
- High Customizability
- High Readability
- Simplicity
- Licensing Friendly 

## Installation

```sh
$ python -m venv venv
$ pip install -r requirements.txt
```


## Usage

```sh
$ cd data ; python generate.py ; cd ..
$ python train.py --config configs/default.yaml
$ python infer.py --model state_dict.pth --input sample_data/*.png
```


## Dependencies and License
....
