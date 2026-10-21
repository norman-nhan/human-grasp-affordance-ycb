# YCB-Affordance Dataset

<img src='images/ycb_affordance_dataset.png' align="right">

We release a large-scale dataset of manually annotated grasps on the 58 objects of the YCB Benchmark Set. This contain grasps such as those in the following image, which are not found by Graspit Simulator or any other automatic pipeline. For each grasp, we annotate the hand position, the hand pose and the grasp type according to the grasp taxonomy of Feix et al. We transfer these grasps to the 92 Video sequences from the YCB-Video dataset, and remove those grasps that are not feasible, due to interpenetrations with other objects or with the table. The dataset contains 133.936 frames, with more than 28M of realistic grasps.

## Prerequisites

Prerequisites are only needed for visualization. You can download the dataset ...

- Python requirements. Run `pip install -r requirements.txt`.

- **MANO layer**: Follow instructions from the MANO layer project in [here](https://raw.githubusercontent.com/hassony2/manopth).

- Copy MANO_LEFT.pkl and MANO_RIGHT.pkl in ./data/mano/

## Download data

- **YCB object set CAD models**: Models from the YCB object set can be downloaded from [here](https://drive.google.com/open?id=1FdAWKpZTJBYctLNOZmlXGP7FGhE4etf0)

- **YCB-Affordance grasps**: Available from [here](https://drive.google.com/file/d/1p_cLNYHIaTAWXJCYENNmKi6Rwmvdzg8x/view?usp=sharing)

- OPTIONAL. **YCB-Video Dataset**: This provides the +133k multi-object scene images that combine with +28M grasps. If you are only interested in the grasps for the YCB object set CAD models, you don't need these. The YCB-Video Dataset can be downloaded from [here](https://drive.google.com/file/d/1if4VoEXNx9W3XCn0Y7Fp15B4GpcYbyYi/view?usp=sharing)


Downloaded data should go in ./data/ following:

```
data/
    models
    grasps
    YCB_Affordance_grasps/
    YCB_Video_Dataset/
        cameras/
        data/
        image_sets/
        ...
```

## Visualization of grasps

### Load objects and grasps on CAD models

With the models and the grasps annotated on the cad models, these can be visualized using the following command.

```
python -u visualize_grasps.py
```

### Load grasps on the YCB-Video dataset

With the YCB Video Dataset downloaded, the grasps on scene can be visualized using the following command. This picks one of the valid grasps at random at plots the objects, the hands in grasping positions, and the plane of the supporting table.

PRESS ANY KEYS TO SEE THE NEXT EXAMPLES
```
python -u visualize_YCB_Affordance.py
```

## Export object-surface human grasp affordances

The original annotations are posed MANO hand meshes, not point-wise object
labels. `export_human_grasp_affordance.py` converts them into deterministic,
point-aligned proximity labels for each of the 58 annotated YCB objects. Run it
from the same `ycb_affordance` environment used by `visualize_grasps.py`:

```
python -u export_human_grasp_affordance.py
```

For a quick single-object check:

```
python -u export_human_grasp_affordance.py \
  --objects 006_mustard_bottle --points 2048 --overwrite
```

Outputs are written under `outputs/human_grasp_affordance/<object>/`:

- `affordance.npz`: aligned points, normals, soft proximity scores, per-grasp
  contact frequency, hard mask, contact count, and minimum hand distance.
- `affordance.ply`: colored preview with the same scalar labels as PLY vertex
  properties.
- `metadata.json`: thresholds, source grasps, taxonomy counts, units, hashes,
  and the exact label definition.
- `manifest.json`: resumable whole-dataset status and per-object counts.

The default hard mask marks object points within 5 mm of at least one annotated
MANO surface. The soft score is a 10 mm Gaussian of the minimum distance. These
are proximity-derived pseudo-contact labels, not measured pressure or force.
Use `--contact-threshold-mm` and `--sigma-mm` to change them. Existing objects
are skipped unless `--overwrite` is supplied.

To overwrite an object's aggregate using only selected annotations, pass exact
pickle paths or a quoted glob. For the manually verified `025_mug` handle
grasps `-5` through `-1`:

```
python -u export_human_grasp_affordance.py \
  --grasp-paths 'data/grasps/obj_21_grasp_-[1-5].pickle' \
  --overwrite
```

This replaces only `outputs/human_grasp_affordance/025_mug`; the other object
exports remain unchanged. The five selected filenames are recorded in its
`metadata.json` and the resolved paths are recorded in the manifest.

## Visualize human grasp affordance results
```
meshlab \path\to\object.ply
```

## Citing

If this dataset is useful in your research, please cite:

```
@inproceedings{corona2020ganhand,
  title={Ganhand: Predicting human grasp affordances in multi-object scenes},
  author={Corona, Enric and Pumarola, Albert and Alenya, Guillem and Moreno-Noguer, Francesc and Rogez, Gr{\'e}gory},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5031--5041},
  year={2020}
}
```

## License

The YCB-Affordance dataset is released only for research purposes
