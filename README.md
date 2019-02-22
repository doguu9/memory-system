# Memory System Info
## Required repositories
[Siamese Network](https://github.com/BerkeleyAutomation/perception/tree/dev_dmwang)

[SD Mask R-CNN](https://github.com/BerkeleyAutomation/sd-maskrcnn)

[Process Wisdom Dataset](https://github.com/BerkeleyAutomation/perception/blob/dev_dmwang/tools/process_wisdom_dataset.py)

[Wisdom Dataset](https://berkeley.box.com/shared/static/7aurloy043f1py5nukxo9vop3yn7d7l3.rar)
## Instructions
All classification tests have been run on the siamese network listed above. All data passed into this network must first be converted to featurized numpy arrays via the script `featurize_images.py` located in the repository. Parameters can be modified in the `train.yaml` file. All grayscale images used for testing have been converted via the `mogrify -type Grayscale [image]` terminal command.

A slightly modified version of the siamese network is included in this repository. The main difference between it and the original is the addition of code allowing for data to be extracted from the training process for error analysis. A corresponding error analysis script for this version of the siamese network is included as well.

All automatic segmentations are produced with the SD Mask R-CNN code listed above. The repository is well-documented and nothing was changed for the memory experiments. Once the segmasks are determined by Mask R-CNN, they need to be processed into individual png files via the Process Wisdom Dataset script. For this portion, depending on the data you are passing in you may need to remove additional padding, rescale, or modify dimensions of the segmasks or images for this portion to correctly segment the images. You can easily check whether or not segmentation is working properly by viewing individual png files put out by the script.

The isolated script for comparing objects via a nearest neighbors algorithm is included in the repository, along with a rudimentary version of the siamese network that passes validation data through the nearest neighbors script before making predictions.

All data used for these experiments is derived from the Wisdom Dataset. A link to download the dataset is listed above. The folder should include the unprocessed RGB and depth images as well as hand-segmented images. Simulated depth images can be found at `/nfs/diskstation/projects/mech_search/siamese_net_training/phoxi_training_sim_dataset`.
