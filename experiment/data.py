import numpy as np
import pandas as pd
import rasterio as rio
from torch.utils.data import Dataset


def get_image(paths, scale, **kwargs):
    """
    Open image with rasterio.
    If list of paths, opens and concatenates on first axis.
    CAREFUL! If kwargs['indexes'] = <int>, will concat on height channel.
    Instead, use kwargs['indexes'] = list(<int>) or add axis to np.concatenate.
    """
    if isinstance(paths, list):
        img = np.concatenate([
            rio_open_image(pth, **kwargs)
            for pth in paths
        ])
    else:
        img = rio_open_image(paths, **kwargs)
    img = np.moveaxis(img, 0, -1)

    return img.astype(float) / scale


def rio_open_image(pth, **kwargs):
    with rio.open(pth) as src:
        return src.read(**kwargs)


class AlaCarteDataset(Dataset):
    def __init__(self, json_path, kind, xcol, ycol,
                 scale_x=255, scale_y=1, transform=None):
        """
        json_path: Path to json with 3 columns named "kind", xcol, ycol
        kind:      Only includes rows json where "kind" matches kind, e.g. train or valid
        xcol:      Path to input image
        ycol:      Path or list of paths to targets (will concat list of images)
        scale_x:   input image will be img_x / scale_x
        scale_y:   input image will be concat(img_y) / scale_y
        transform: (composition of) transform(s) applied to pair (img, tgt)
        """
        self.df = pd.read_json(json_path)
        self.df = self.df.loc[self.df["kind"] == kind].reset_index(drop=True)
        self.xcol = xcol
        self.ycol = ycol
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.transform = transform

    def __getitem__(self, idx):
        img = get_image(self.df.loc[idx, self.xcol], scale=self.scale_x)
        tgt = get_image(self.df.loc[idx, self.ycol], scale=self.scale_y)

        if self.transform:
            img, tgt = self.transform(img, tgt)

        return img, tgt

    def __len__(self):
        return len(self.df)
