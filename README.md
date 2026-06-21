Adversarial Transformer Networks for High-Fidelity Image Style Transfer
Authors: [TUNG-THIEN LAM](https://github.com/rogerlamfullstack), NAEEM UL ISLAM, CHENG-ZEN YANG

## Experiment
### Requirements
* python 3.10
* pytorch 2.1.2
* PIL, numpy, scipy
* tqdm  <br> 


# Training  
## Pretrained Models

Download the pretrained models from the following links:

-   [VGG
    model](https://drive.google.com/file/d/1BinnwM5AmIcVubr16tPTqxMjUCE8iu5M/view?usp=sharing)
-   [ViT
    embedding](https://drive.google.com/file/d/1C3xzTOWx8dUXXybxZwmjijZN8SrC3e4B/view?usp=sharing)
-   [Decoder](https://drive.google.com/file/d/1fIIVMTA_tPuaAAFtqizr6sd1XV7CX6F9/view?usp=sharing)
-   [Transformer
    module](https://drive.google.com/file/d/1dnobsaLeE889T_LncCkAA2RkqzwsfHYy/view?usp=sharing)

Place all downloaded files into the `./pretrain/` directory.

------------------------------------------------------------------------

## Datasets

### Style Dataset

We use WikiArt collected from https://www.wikiart.org/.

The style categories used in this work include: Action Painting,
Analytical Cubism, Early Renaissance, Fauvism, High Renaissance,
Mannerism, Naive Art, New Realism, Northern Renaissance, Pointillism,
Pop Art, Rococo, and Synthetic Cubism.

### Content Dataset

We use the COCO 2014 train split from the COCO dataset:
https://cocodataset.org/#download

```
python3.10 train.py train_test_example_10000 --style_dir datasets/style --content_dir dataset/train2014 --save_dir output --batch_size 2 --strategy_step 10000
```

## Testing 
### Test with Image
Generate stylized images using the content and style images as input:
```
python test.py  --content_dir input/content/ --style_dir input/style/    --output out
```

### Test with Metrics
Evaluate the generated images using SSIM, PSNR, and LPIPS:
```
python3.10 test_with_metrics.py  --content_dir input/content/ --style_dir input/style/
```
After generating the CSV results, compute the average metric scores with:
```
python3 eval.py # put csv files to metrics folder
```


### Reference
This project is built upon StyTR². Please refer to the original implementation:
https://github.com/diyiiyiii/StyTR-2
```
@inproceedings{deng2021stytr2,
      title={StyTr^2: Image Style Transfer with Transformers}, 
      author={Yingying Deng and Fan Tang and Weiming Dong and Chongyang Ma and Xingjia Pan and Lei Wang and Changsheng Xu},
      booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2022},
}
```