## Copyright (c) Microsoft Corporation.
## Licensed under the MIT license.

## This code is modified by Linjie Li from Hengyuan Hu's repository.
## https://github.com/hengyuan-hu/bottom-up-attention-vqa
## GNU General Public License v3.0

## Script for downloading data

# VQA Questions
wget -P data https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip
unzip data/v2_Questions_Train_mscoco.zip -d data/Questions
rm data/v2_Questions_Train_mscoco.zip

wget -P data https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip
unzip data/v2_Questions_Val_mscoco.zip -d data/Questions
rm data/v2_Questions_Val_mscoco.zip

wget -P data https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip
unzip data/v2_Questions_Test_mscoco.zip -d data/Questions
rm data/v2_Questions_Test_mscoco.zip

# VQA Annotations
wget -P data https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip
wget -P data https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip
unzip data/v2_Annotations_Train_mscoco.zip -d data/Answers
rm data/v2_Annotations_Train_mscoco.zip
unzip data/v2_Annotations_Val_mscoco.zip -d data/Answers
rm data/v2_Annotations_Val_mscoco.zip

# VQA cp-v2 Questions
mkdir data/cp_v2_questions
wget -P data/cp_v2_questions https://computing.ece.vt.edu/~aish/vqacp/vqacp_v2_train_questions.json 
wget -P data/cp_v2_questions https://computing.ece.vt.edu/~aish/vqacp/vqacp_v2_test_questions.json 

# VQA cp-v2 Annotations
mkdir data/cp_v2_annotations
wget -P data/cp_v2_annotations https://computing.ece.vt.edu/~aish/vqacp/vqacp_v2_train_annotations.json 
wget -P data/cp_v2_annotations https://computing.ece.vt.edu/~aish/vqacp/vqacp_v2_train_annotations.json

# Visual Genome Annotations
mkdir data/visualGenome
wget -P data/visualGenome https://convaisharables.blob.core.windows.net/vqa-regat/data/visualGenome/image_data.json
wget -P data/visualGenome https://convaisharables.blob.core.windows.net/vqa-regat/data/visualGenome/question_answers.json

# GloVe Vectors and dictionary
wget -P data https://convaisharables.blob.core.windows.net/vqa-regat/data/glove.zip
unzip data/glove.zip -d data/glove
rm data/glove.zip

# Image Features
# adaptive
# WARNING: This may take a while
mkdir data/Bottom-up-features-adaptive
wget -P data/Bottom-up-features-adaptive https://convaisharables.blob.core.windows.net/vqa-regat/data/Bottom-up-features-adaptive/train.hdf5
wget -P data/Bottom-up-features-adaptive https://convaisharables.blob.core.windows.net/vqa-regat/data/Bottom-up-features-adaptive/val.hdf5
wget -P data/Bottom-up-features-adaptive https://convaisharables.blob.core.windows.net/vqa-regat/data/Bottom-up-features-adaptive/test2015.hdf5

# fixed
# WARNING: This may take a while
mkdir data/Bottom-up-features-fixed
wget -P data/Bottom-up-features-fixed https://convaisharables.blob.core.windows.net/vqa-regat/data/Bottom-up-features-fixed/train36.hdf5
wget -P data/Bottom-up-features-fixed https://convaisharables.blob.core.windows.net/vqa-regat/data/Bottom-up-features-fixed/val36.hdf5
wget -P data/Bottom-up-features-fixed https://convaisharables.blob.core.windows.net/vqa-regat/data/Bottom-up-features-fixed/test2015_36.hdf5

# imgids
wget -P data/ https://convaisharables.blob.core.windows.net/vqa-regat/data/imgids.zip
unzip data/imgids.zip -d data/imgids
rm data/imgids.zip

# Download Pickle caches for the pretrained model
# and extract pkl files under data/cache/.
wget -P data https://convaisharables.blob.core.windows.net/vqa-regat/data/cache.zip
unzip data/cache.zip -d data/cache
rm data/cache.zip

# Download pretrained models
# and extract files under pretrained_models.
wget https://convaisharables.blob.core.windows.net/vqa-regat/pretrained_models.zip
unzip pretrained_models.zip -d pretrained_models/
rm pretrained_models.zip
