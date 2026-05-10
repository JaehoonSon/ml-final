"""Download Pascal VOC 2012 segmentation data via torchvision."""

from torchvision.datasets import VOCSegmentation

ROOT = "data/voc"


def main():
    print(f"Downloading Pascal VOC 2012 to {ROOT} ...")
    VOCSegmentation(root=ROOT, year="2012", image_set="train", download=True)
    VOCSegmentation(root=ROOT, year="2012", image_set="val", download=True)
    print("Done.")


if __name__ == "__main__":
    main()
