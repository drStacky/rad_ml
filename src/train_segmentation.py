from contextlib import contextmanager
import shutil
import tempfile

import click
import boto3
import numpy as np
from pathlib import Path
import pytorch_lightning as pl
from urllib.parse import urlparse
import yaml

import data
import pl_module
import transforms as T


@contextmanager
def temporary_directory():
    """
    Create temporary directory
    """
    name = tempfile.mkdtemp()
    try:
        yield name
    finally:
        shutil.rmtree(name)


def _parse_s3_path(s3_path):
    """
    Parse s3 bucket and key from path
    :param s3_path: Path on s3
    """
    parsed = urlparse(s3_path)
    if parsed.scheme != 's3':
        raise ValueError(f'input path "{s3_path}" does not exist locally'
                         ' or is not of the form "s3://bucket/key"')

    bucket_name, key = parsed.netloc, parsed.path[1:]
    if not bucket_name:
        raise ValueError('no valid bucket name parsed from {s3_path}')
    return bucket_name, key


def get_local_pth(pth, scratch_dir, download=False):
    """
    Download to scratch_dir and return local path if in s3.
    """
    if pth.startswith('s3://'):
        bucket, key = _parse_s3_path(pth)
        
        local_pth = str(Path(scratch_dir).resolve()/Path(key).name)

        aws_s3 = boto3.resource('s3')
        if download:
            aws_s3.Object(bucket, key).download_file(local_pth)
    else:
        local_pth = pth

    return local_pth


def read_yaml(pth):
    """
    Read a local yaml and return dict of contents
    """
    with open(pth, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None


@click.command()
@click.argument('yml_path')
def main(yml_path):
    with temporary_directory() as scratch_dir:
        local_yml_path = get_local_pth(yml_path, scratch_dir, download=True)
        config = read_yaml(local_yml_path)
        local_json_pth = get_local_pth(config['json_pth'], scratch_dir, download=True)
        local_log_dir = get_local_pth(config['log_dir'], scratch_dir)

        # Initialize training classes
        local_bs = 8
        total_bs = config.get('batch_size', local_bs)
        nw = 8
        epochs = config.get('num_epochs', 1)
        
        # Transforms and Augmentations
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        tfms = [T.ToTensor(), T.Normalize(mean, std)]
        val_tfms = T.Compose(tfms)
        tfms = [T.RandomDihedral()] + tfms
        trn_tfms = T.Compose(tfms)

        # Datasets
        trn_ds = data.SegDataset(
            local_json_pth, 'train',
            'img_path', 'lbl_path',
            scale_x=255, scale_y=1,
            transform=trn_tfms,
        )
        val_ds = data.SegDataset(
            local_json_pth, 'valid',
            'img_path', 'lbl_path',
            scale_x=255, scale_y=1,
            transform=val_tfms,
        )

        tst_ds = data.SegDataset(
            local_json_pth, 'test',
            'img_path', 'lbl_path',
            scale_x=255, scale_y=1,
            transform=val_tfms,
        )

        # Metrics
        # TODO This will have to be custom
        # metrics = [IoU(num_classes=2), Precision(num_classes=2), Recall(num_classes=2)]
        metrics = []
        
        fcn = pl_module.FCNSegmentation(
            pretrained=True,
            num_classes=1,
            trn_ds=trn_ds,
            val_ds=val_ds,
            tst_ds=tst_ds,
            bs=local_bs, nw=nw,
            metrics=metrics,
        )

        trainer = pl.Trainer(
            gpus=1,
            max_epochs=epochs,
            accumulate_grad_batches=total_bs // local_bs,
            benchmark=True,
            num_sanity_val_steps=2,
            limit_train_batches=config.get('trn_pct', 1.),
            limit_val_batches=config.get('val_pct', 1.),
            limit_test_batches=config.get('tst_pct', 1.),
            logger=pl.loggers.TensorBoardLogger(local_log_dir,
                                                default_hp_metric=False),
        )

        # Train and test
        trainer.fit(fcn)
        trainer.test()

        # Move logs to S3
        bucket, key = _parse_s3_path(config['log_dir'])
        boto3.resource('s3').Bucket(bucket)\
                            .upload_file(Filename=local_log_dir,
                                         Key=key,
                                         ExtraArgs={'ACL': 'bucket-owner-full-control'})
        

if __name__ == "__main__":
    main()
