# Modeling Missing Annotations for Incremental Learning in Object Detection -- @CVPR 22, [CLVision Workshop](https://sites.google.com/view/clvision2022) [Paper](https://arxiv.org/abs/2204.08766)

## Official PyTorch implementatation based on Detetron (v1) - Fabio Cermelli, Antonino Geraci, Dario Fontanel, Barbara Caputo

![teaser](https://user-images.githubusercontent.com/10979083/169845945-0e086da0-a26b-4486-8d61-ffc7335287b6.png)

Despite the recent advances in the field of object detection, common architectures are still ill-suited to incrementally detect new categories over time. They are vulnerable to catastrophic forgetting: they forget what has been already learned while updating their parameters in absence of the original training data. Previous works extended standard classification methods in the object detection task, mainly adopting the knowledge distillation framework. However, we argue that object detection introduces an additional problem, which has been overlooked. While objects belonging to new classes are learned thanks to their annotations, if no supervision is provided for other objects that may still be present in the input, the model learns to associate them to background regions. We propose to handle these missing annotations by revisiting the standard knowledge distillation framework. Our approach outperforms current state-of-the-art methods in every setting of the Pascal-VOC dataset. We further propose an extension to instance segmentation, outperforming the other baselines.

![method](https://user-images.githubusercontent.com/10979083/169845969-94d94e57-19f9-45bd-81c5-7c77f21ad55e.png)

# How to run
## Install
Please, follow the instruction provided by Detectron 1 and found in install.md

## Dataset
You can find the Pascal-VOC dataset already in Detectron. Pascal-SBD, used in instance segmentation, can be found in the `data` folder.

## Run!
We provide scripts to run the experiments in the paper (FT, ILOD, Faster-ILOD, MMA and ablations).
You can find three scripts: `run.sh`, `run-ms.sh`, and `run-is.sh`. The file can be used to run, respectively: single-step detection settings (19-1, 15-5, 10-10), multi-step detection settings (10-5, 10-2, 15-1, 10-1), and instance segmentation settings (19-1, 15-5).

Without specifying any option, the defaults will load the ILOD method using the Faster-RCNN. In particular, it will only introdiuce the L2 distillation loss on the normalized class scores.
You can play with the following parameters to obtain all the results in the paper:
- `--rpn` will introduce the feature distillation (default: not use);
- `--feat` with options [`no`, `std`, `align`, `att`]. No means not using feature distillation (as in MMA), while `std` is the feature distillation employed in Faster-ILOD (default: no).
- `--uce` will enable the unbiased classification loss (Eq. 3 of the paper) - (default: not use);
- `--dist_type` with options [`l2`, `uce`, `ce`, `none`], where l2 is the distillation used in ILOD, UCE the one used in our method MMA, CE is used in ablation study, and none means not use it (default: l2);
- `--cls` is a float indicating the weight of the distillation loss (default: 1.); In MMA we used different parameters for cls depending on the number of learned classes: 1. when learning 1 or 2 classes, 0.5 when learning 5, 0.1 when learning 10;
- `--mask` is a float indicating the weight of the mask distillation loss and it's used only on instance segmentation (default: 1.). We used 0.5 for our improved version of MMA.

# Cite us!
``` 
@InProceedings{cermelli2022modeling,
  title={Modeling Missing Annotations for Incremental Learning in Object Detection},
  author={Fabio Cermelli, Antonino Geraci, Dario Fontanel, Barbara Caputo},
  booktitle={In Proceedings of the IEEE/CVF Computer Vision and Patter Recognition Conference (CVPR), CLVISION workshop},
  month={June},
  year={2022}
}
```

# Acknowledgments
Our repository is based on the amazing work of @CanPeng123 [FasterILOD](https://github.com/CanPeng123/Faster-ILOD) and on the [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) library. We thank the authors and the contibutors of these projects for releasing their code.
