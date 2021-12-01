# GitHub for "A learning based framework for disease prediction from images of human-derived pluripotent stem cells of schizophrenia patients"

The code for this paper is located in the bin/Notebooks folder, and each classifier has its own folder.
The remaining python files in this folder are preprocessing steps used to generate the images, contained in the "data" folder and "Rep_Imgs" folder, while also generating csv files for the radiomics features in the "r" folder, from the raw data stored in the "Images" folder.


## Sample commands for running the Convolutional Neural Network(CNN):
First, navigate to the NeuronCNNGrouping folder, then run a command similar to the following examples listed below, where the user can vary the arguments as desired.

1) python main.py --data_subset HC --in_channels 1 --nbands 1 --channels FGF --patch_size 16 --patch_stride 8 --num_runs 10 --num_epochs 20

2) python main.py --data_subset CHIR --in_channels 4 --nbands 4 --channels TUBB FGF MAP NAV --patch_size 16 --patch_stride 8 --num_runs 10 --num_epochs 20

The data_subset parameter chooses whether to classify for diagnosis or classify for perturbation, e.g. CHIR will classify for diagnosis in the GSK3i inhibited cells while HC will classify for perturbation.

The parameters in_channels and nbands should be the same and determine the number of image channels to use as input; the channels parameter chooses the specific channels to use. For example, the first command uses only the FGF channel while the second command uses all four channels.

The patch_stride parameter chooses how much over lap should occur between patches while the patch_size parameter chooses the size of the patches.

The remaining two parameter's choose how many times to run the classifier -num_runs- and how many epochs should occur during each run -num_epochs.

For the full list of tunable parameters, see the args.py file.

## Sample commands for running the Vision Transformer(ViT):
As with the CNN, navigate to the ViT folder, and then run a command such as one of the following.

1) python main.py --data_subset SCZ --in_channels 1 --nbands 1 --channels MAP --num_runs 10 --num_epochs 20

2) python main.py --data_subset DMSO --in_channels 4 --nbands 4 --channels TUBB FGF MAP NAV --num_runs 2 --num_epochs 5

The parameter definitions are similar to the CNN parameters and can be referenced in the args.py file.

## Instructions for running the Support Vector Machine(SVM):
Navigate to the SVM folder then run either of the following commands;

1) python bow_classify_r_417.py
This command will perform classification using two concatenated features such as FGF and TUBB. The user chooses which features to use by changing the two input csv filenames loaded in the python script.

2) python bow_classify_r_417_4chan.py
This command will perform classification using all four concatenated features.

3) python bow_classify_r_417_not_concat.py
This command will perform classification using only a single channel. The user selects which channel to use by modifying the radiomics csv file loaded in the python script.

Note, the save_names for the output performance csv files should be changed according to the input channel. This applies to both the first and third SVM python scripts.

The bag of words(bow) features used for the support vector machine are derived from the gray-level-dependence matrix(GLDM). For more information, please read the paper and check the pyradiomics documentation.
