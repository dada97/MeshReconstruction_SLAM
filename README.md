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
└── slamData.csv
```

## Output Data
```bash
./[output_dir]/outputMesh.obj
```

## Test Example
```bash
MeshReconstruction.exe -i [input_dir] -o [output_dir] -d [boolean]
```