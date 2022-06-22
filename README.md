# RoadMeshReconstruction
this project run on Visual Studio 2019

## Third Party Libraries
- Opencv
- CGAL

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
## Output Data structure
```bash
.
├── output.obj
└── pointcloud.ply
```

## Command Line
```bash
MeshReconstruction.exe [input_dir] [output_dr]
```