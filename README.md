# mldl_rgb-d_da_project
Project for a different implementation of https://github.com/MRLoghmani/relative-rotation for DA on RGB-D pictures.

## For a strong insight in the project, please read "Report.pdf" available in the repository folder.

To run the code:

1) create a dataset folder, check check_project.sh to see how to create it
2) if running on linux, run the check_project.sh or run.sh script in the same folder of the python scripts

Paper based on:

1) https://arxiv.org/pdf/1909.11825.pdf suggesting to implement multiple tasks for recognition -> https://github.com/yueatsprograms/uda_release
2) https://openreview.net/pdf?id=BBrl21Hw--q reinforce point 1 as a good route
3) https://arxiv.org/pdf/1603.05027v3.pdf paper for changing the layers position


New Implementation:

1) Changing implementation on original pytorch https://github.com/pytorch/vision/blob/1aef87d01eec2c0989458387fa04baebcc86ea7b/torchvision/models/resnet.py#L75
2) Code for the bottleneck has been modified from https://docs.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-train-model .
3) https://github.com/bbrattoli/JigsawPuzzlePytorch
