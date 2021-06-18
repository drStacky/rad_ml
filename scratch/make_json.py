import json
import random

#import boto3


def get_kind(key):
    test_cid = '10400100099FE500'
    if test_cid in key:
        return 'test'
    elif random.random() < 0.1:
        return 'valid'
    else:
        return 'train'


random.seed(0)

"""
# Retrieve the list of existing buckets
dev = boto3.session.Session(profile_name='rdml')
s3 = dev.resource('s3')
bucket = s3.Bucket('mstackpo')

data = [
    {
        'img_path': f's3://{bucket.name}/{obj.key}',
        'lbl_path': f's3://{bucket.name}/{obj.key.replace("img", "build")}',
        'kind': get_kind(obj.key),
    }
    for obj in bucket.objects.filter(Prefix='raster_chips/img/')
    if obj.size
]

with open('/Users/mattstackpo/PycharmProjects/rad_ml_aws/experiment/shanghai_build.json', 'w') as f:
"""

from pathlib import Path

data = [
    {
        'img_path': str(pth),
        'lbl_path': str(pth).replace('img', 'msk'),
        'kind': get_kind(str(pth)),
    }
    for pth in Path('/mnt/tier5/mstackpole/rad_ml/data_sample/img').glob('*.tif')
]

with open('/mnt/tier5/mstackpole/rad_ml/data_sample/data.json', 'w') as f:
    json.dump(data, f, indent=3)
