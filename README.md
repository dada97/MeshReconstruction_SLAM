# RoadMeshReconstruction
## Overview
![Alt text](./assets/RoadMesh.png?raw=true "Title")

## Visual studio Version
this project build on Visual Studio 2019

## Third Party Libraries
- Opencv 3.4.0
- CGAL 5.4
- Boost 1.71
- Eigen 3.4

## Input Data structure
```bash
.
├── rgb
│   ├── 0.jpg
│   ├── 1.jpg
│   └── ...
├── seg
│   ├── 0_prediction.png
│   ├── 1_prediction.png
│   └── ... 
├── depth
│   ├── 0_disp.npy
│   ├── 1_disp.npy
│   └── ...
├── slamData.csv
└── landmarks.csv
```

## Output Data
```bash
.
├── outputMesh.obj
└── scale.txt
```

## Video to Image Sequence

Example: python videoToImg.py ./video.mp4 ./rgb
```bash
python videoToImg.py [input_path].mp4 [output_dir]
```

## SLAM Data Preprocess

Example: python msgToCSV.py.py ./map.msg ./

output: 

[output_path]/slamData.csv

[output_path]/landmarks.csv
```bash
python map.py [input_path].msg [output_path]
```

## Test Example
```bash
MeshReconstruction.exe -i [input_dir] -o [output_dir] -d [boolean]
```