###Part3




### 1. Prepare the datasets.


- (1) SYSU-MM01 Dataset [4]: The SYSU-MM01 dataset can be downloaded from this [website](http://isee.sysu.edu.cn/project/RGBIRReID.htm).

   - run `python pre_process_sysu.py` to pepare the dataset, the training data will be stored in ".npy" format.



### 2. Training.
Train a model by:
```
python train.py --dataset sysu --gpu 1
```


--gpu: which gpu to run.

You may need mannully define the data path first.

Parameters: More parameters can be found in the script.

### 3. Testing.
Test a model on  SYSU-MM01 dataset by
```
python test.py --mode all --tvsearch True --resume 'model_path' --gpu 1 --dataset sysu
```
--mode: "all" or "indoor" all search or indoor search (only for sysu dataset).







### 5. Citation
Our code is referenced by this work:

```
@InProceedings{Zhang_2023_CVPR,
    author    = {Zhang, Yukang and Wang, Hanzi},
    title     = {Diverse Embedding Expansion Network and Low-Light Cross-Modality Benchmark for Visible-Infrared Person Re-Identification},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {2153-2162}
}
```


### 6. Contact

If you have any question, please feel free to contact us. zhangyk@stu.xmu.edu.cn.
