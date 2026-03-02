# "Learning Semantic Latent Directions for Accurate and Controllable Human Motion Prediction" (**ECCV 2024**)

<img src="images/intro.png" width="100%"/>

---
This repo contains the official implementation of the paper:

Learning Semantic Latent Directions for Accurate and Controllable Human Motion Prediction

ECCV 2024
[[arxiv](https://arxiv.org/abs/2407.11494)]
### Dependencies
* Python >= 3.8
* [PyTorch](https://pytorch.org) >= 1.9
* Tensorboard
* matplotlib
* tqdm
* argparse

### Get the data
We adapt the data preprocessing from [GSPS](https://github.com/wei-mao-2019/gsps).
* We follow the data preprocessing steps ([DATASETS.md](https://github.com/facebookresearch/VideoPose3D/blob/master/DATASETS.md)) inside the [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) repo.
* Given the processed dataset, we further compute the multi-modal future for each motion sequence. All data needed can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1sb1n9l0Na5EqtapDVShOJJ-v6o-GZrIJ?usp=sharing) and place all the dataset in ``data`` folder inside the root of this repo.

### Get the pretrain models
* All pretrain models can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1YAa3Lpei0V3-JTEZwSw0WqRfSTYbY-z2?usp=drive_link) and place all the pretrain models in ``results`` folder inside the root of this repo.

### Train
We have used the following commands for training the network on Human3.6M or HumanEva-I with skeleton representation:
```bash
python train_nf.py --cfg [h36m/humaneva] --gpu_index 0
python main.py --cfg [h36m/humaneva] --gpu_index 0
```
 ### Test
 To test on the pretrained model, we have used the following commands:
  ```bash
 python main.py --cfg [h36m/humaneva] --mode test --iter 500 --gpu_index 0
  ```
 ### Visualization
 For visualizing from a pretrained model, we have used the following commands:

   ```bash
 python main.py --cfg [h36m/humaneva] --mode viz --iter 500 --gpu_index 0
  ```

### Model Params / FLOPs / Inference Time (table file)
You can use `profile_model_table.py` to read an input CSV table and append:
- Params (M)
- FLOPs (G)
- Inference Time (ms / fps)

1) Prepare an input CSV file (for example `profile_input.csv`, or use `profile_input_template.csv`):
```csv
model_name,cfg,batch_size
ours_h36m,h36m,1
ours_humaneva,humaneva,1
```

2) Run:
```bash
python profile_model_table.py \
  --input-table profile_input.csv \
  --output-table profile_output.csv \
  --device auto \
  --input-mode history \
  --warmup 20 \
  --repeat 50
```

3) The output CSV will append columns such as:
`Params(M)`, `FLOPs(G)`, `Inference Time(ms)`, `Inference Speed(fps)`.

Notes:
- `input-mode history` matches test-time usage in `main.py` (`model(X)` with only history frames).
- FLOPs is computed with `thop` if available; otherwise it falls back to `torch.profiler`.

 ### Acknowledgments
 
 This code is based on the implementations of [STARS](https://github.com/Sirui-Xu/STARS).

 ## Citation
If you find this work useful in your research, please cite:

```bibtex
@article{xu2024learning,
  title={Learning Semantic Latent Directions for Accurate and Controllable Human Motion Prediction},
  author={Xu, Guowei and Tao, Jiale and Li, Wen and Duan, Lixin},
  journal={arXiv preprint arXiv:2407.11494},
  year={2024}
}
```

## License

This repo is distributed under an [MIT LICENSE](LICENSE)
