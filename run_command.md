# Train Classifier on the Entire Dataset

```
python train_imagenet.py --epochs 90 --lr 0.1 --scheduler cosine --task-name imagenet --base-dir ./traj --data-dir ../tiny-imagenet-200 --network resnet34 --batch-size 256 --gpuid 0 --num-workers 0
```
