"""
Reorganize the Tiny ImageNet validation split into ImageFolder format.

Tiny ImageNet ships its validation split as:
    val/images/*.JPEG  +  val/val_annotations.txt
which is NOT the per-class layout torchvision's ImageFolder expects. This script
rearranges it into:
    val/<wnid>/*.JPEG
so ImageFolder (and train_imagenet.py) can read it.

Idempotent: if there is no val/images dir or no val_annotations.txt, it does nothing.

Usage:
    python reorganize_val.py --val-dir ../tiny-imagenet-200/val
"""
import os
import shutil
import argparse


def reorganize_tiny_imagenet_val(val_dir):
    images_dir = os.path.join(val_dir, 'images')
    annotations = os.path.join(val_dir, 'val_annotations.txt')

    if not (os.path.isdir(images_dir) and os.path.isfile(annotations)):
        print(f'Nothing to do: {val_dir} is already reorganized (or not Tiny ImageNet).')
        return

    print(f'Reorganizing Tiny ImageNet val folder into per-class subdirs: {val_dir}')
    moved = 0
    with open(annotations, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            filename, wnid = parts[0], parts[1]
            class_dir = os.path.join(val_dir, wnid)
            os.makedirs(class_dir, exist_ok=True)
            src = os.path.join(images_dir, filename)
            dst = os.path.join(class_dir, filename)
            if os.path.isfile(src):
                shutil.move(src, dst)
                moved += 1

    # remove the now-empty images dir so ImageFolder doesn't treat it as a class
    if os.path.isdir(images_dir) and not os.listdir(images_dir):
        os.rmdir(images_dir)

    print(f'Done. Moved {moved} images into per-class folders under {val_dir}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reorganize Tiny ImageNet val folder for ImageFolder.')
    parser.add_argument('--val-dir', type=str, default='../tiny-imagenet-200/val',
                        help='Path to the Tiny ImageNet val directory.')
    args = parser.parse_args()
    reorganize_tiny_imagenet_val(args.val_dir)
