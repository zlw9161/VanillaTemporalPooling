### VanillaTemporalPooling
Created by <a href="https://github.com/zlw9161">Liwen Zhang</a> from Speech Lab @ Harbin Institute of Technology, Harbin, China.

### Citation
If you find our work useful in your research, please cite:

        @article{liu:2019:cpnet,
          title={Unsupervised Temporal Feature Learning Based on Sparse Coding Embedded BoAW},
          author={Liwen Zhang, Jiqing Han and Shiwen Deng},
          conference={Interspeech},
          year={2018}
        }

### Abstract
Vanilla Temporal Pooling is an unsupervised temporal feature learning method, which can effectively capture the temporal dynamics for an entire audio signal with arbitrary duration by building direct connections between the BoAW histograms sequence and its time indexes using a non-linear Support Vector Regression (SVR) model. Furthermore, to make the feature representation have a better signal reconstruction ability, we embedded the sparse coding approach in the conventional BoAW framework. 

### Contact
* Liwen Zhang (lwzhang9161@126.com and 15B903062@hit.edu.cn)

### Work Environment
* Matlab 2016b or 2018b

### Dependency
* vlfeat-0.9.20
* liblinear-2.20
* libsvm-3.23
* spams-matlab-v2.6

### Dataset
* Audio Event Recognition dataset can be downloaded at:
https://bitbucket.org/naoya1/aenet_feat/src/master/

### Code Description
* FeatureExtractor_AE
This function is used to extract MFCC features for the entire audio waves;
* SegFeatExtractor_AE
This function is used to extract MFCC features for the audio segments with overlap;
* GenFileList
Generate data list for the training/test data;
* BoWExtractor_AE
Generate BoAW features with MFCC features;
* NNSCFeatExtractor_AE
Generate BoAW_SC (SC for Sparse Coding) features with MFCC features;
* TemporalPooling_AE
Generate audio representations with BoAW/BoAW_SC features;
Train the svm classifier with the audio representations;

### License
Our code is released under our License (see LICENSE file for details).

### Related Projects
* [TASLP 2017 paper - AENet: Learning Deep Audio Features for Video Analysis](http://arxiv.org/pdf/1701.00599) by Naoya Takahashi et al.
* [TPAMI 2016 paper - Rank Pooling for Action Recognition](http://users.cecs.anu.edu.au/~basura/papers/PAMI2016Fernando.pdf) by Fernando Basura et al.
