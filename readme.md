
# Periodically Exchange Teacher-Student for Source-Free Object Detection
- 🔔This is the re-implementation code for the paper "Periodically Exchange Teacher-Student for Source-Free Object Detection", ICCV 2023.
- Sometimes, it's also feasible to apply exchange strategies within a single teacher-student framework to alleviate instability issues. 
![](https://drive.google.com/uc?id=1xPadrtuMUzvY_ghG7f5GHXwIVPyWo13J)
# 🛠️Setup
## Runtime

The main python libraries we use:
- python 3.8
- torch 1.8.1
- [detectron2](https://github.com/facebookresearch/detectron2)

## Datasets
For convenience, the data annotation files have been uniformly processed into JSON files following COCO format. By the way, we have also processed the Clipart1K and Watercolor used in other methods. The source images of the dataset requires an additional download. The dataset configurations can be changed in PETS/ubteacher/data/datasets/builtin.py.

[GoogleDrive Link](https://drive.google.com/drive/folders/1NEBSdhcJtQmMkOGYNPXDkYf9spfSBmuE?usp=drive_link)


## Source Weights

The C2F and C2B tasks share source weights in the method. 

[GoogleDrive Link](https://drive.google.com/drive/folders/1dQ6i2-PPiPuxfsELjD6nU4MUTo8DLrST?usp=drive_link)


# 🎢Run
Use the packaged Docker image if you don't want to build the environment:
```shell
docker pull cosword/mydocker:v1.6.9
```

For example, to run the experiment, just enter the following cmd on root directory:
```shell
python train_net --config configs/UDA/C2F.yaml
```

# 📌Citation
If you would like to cite our works, the following bibtex code may be helpful:
```text
@inproceedings{liu2023periodically,
  title={Periodically exchange teacher-student for source-free object detection},
  author={Liu, Qipeng and Lin, Luojun and Shen, Zhifeng and Yang, Zhifeng},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={6414--6424},
  year={2023}
}
```


# ⚖️License
This source code is released under the MIT license. View it [here](LICENSE)
