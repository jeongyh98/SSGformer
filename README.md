# SSGformer
[ICCV 2025] Robust Adverse Weather Removal via Spectral-based Spatial Grouping

## :mega: Citation
If you use SSGformer, please consider citing:

    @inproceedings{jeong2025robust,
        title={Robust Adverse Weather Removal via Spectral-based Spatial Grouping},
        author={Jeong, Yuhwan and Yang, Yunseo and Yoon, Youngho and Yoon, Kuk-Jin},
        booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
        pages={11872--11883},
        year={2025}
    }
---


## :bell: Visual Results 


All visual results are in [Google Drive](https://drive.google.com/drive/folders/1T7eKWKSe_IlcIF4iFtm9N-zqz0aQdz2r?usp=sharing). 


## :mask: Mask generation

    python ./mask/mask.py --root {path/to/(Raindrop/Rain/Snow)} --save_path {path/to/save/mask}



## :tree: Dataset setting

    allweather_dataset/
    ├── train/
    │   └── allweather/
    │       ├── gt/
    │       ├── input/
    │       └── mask/
    └── test/
        ├── rain_drop_test/
        │   ├── gt/
        │   ├── input/
        │   └── mask/
        ├── Snow100K-L/
        │   ├── gt/
        │   ├── input/
        │   └── mask/
        └── test1/
            ├── gt/
            ├── input/
            └── mask/


## :+1: Acknowledgment
Our code is based on the [Histoformer](https://github.com/sunshangquan/Histoformer).