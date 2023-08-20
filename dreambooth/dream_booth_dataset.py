from diffusers.utils import load_image, PIL_INTERPOLATION, randn_tensor
import torch
import PIL
import numpy as np
import warnings
import os

from pathlib import Path

from torch.utils.data import Dataset

from typing import List, Optional, Union



# copy from diffusers.utils.load_image

def load_image(image: Union[str, PIL.Image.Image]) -> PIL.Image.Image:
    """
    Loads `image` to a PIL Image.

    Args:
        image (`str` or `PIL.Image.Image`):
            The image to convert to the PIL Image format.
    Returns:
        `PIL.Image.Image`:
            A PIL Image.
    """
    if isinstance(image, str):
        image = PIL.Image.open(image)
    else:
        raise ValueError(
            "Incorrect format used for image. Should be an url linking to an image, a local path, or a PIL image."
        )
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image






# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.preprocess
def preprocess(image):
    warnings.warn(
        "The preprocess method is deprecated and will be removed in a future version. Please"
        " use VaeImageProcessor.preprocess instead",
        FutureWarning,
    )
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        w, h = image[0].size
        w, h = 512, 512

        w, h = (x - x % 8 for x in (w, h))  # resize to integer multiple of 8

        image = [np.array(i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)
    return image


class DreamBoothDataset(Dataset):

    def __init__(
            self,
            instance_data_root,
            instance_prompt,
            class_data_root=None,
            class_prompt=None,
            size=512,

            ):
        self.instance_data_root = instance_data_root
        self.instance_prompt = instance_prompt
        self.class_data_root = class_data_root
        self.class_prompt = class_prompt
        self.size = size

        self.instance_image_names = [
                str(Path(self.instance_data_root) / item)
                for item in os.listdir(self.instance_data_root)]

        self.instance_images = [load_image(image_name) for image_name in self.instance_image_names]

        self.class_image_names = []

        if self.class_data_root is not None:
            self.class_image_names = [
                    str(Path(self.class_data_root) / item)
                    for item in os.listdir(self.class_data_root)]

        self.len_instance = len(self.instance_image_names)
        self.len_class = len(self.class_image_names)


    def __len__(self):

        return max(len(self.instance_image_names), len(self.class_image_names))
        


    def __getitem__(self, idx):
        instance_prompt = self.instance_prompt
        # instance_image = preprocess(load_image(self.instance_image_names[idx % self.len_instance]))

        # instance_image = load_image(self.instance_image_names[idx % self.len_instance])
        instance_image = self.instance_images[idx % self.len_instance]

        class_prompt, class_image = None, None

        if self.len_class > 0:
            class_prompt = self.class_prompt
            # class_image = preprocess(load_image(self.class_image_names[idx % self.len_class]))
            class_image = load_image(self.class_image_names[idx % self.len_class])

        sample = dict(
            instance_prompt=instance_prompt,
            instance_image=instance_image,
            class_prompt=class_prompt,
            class_image=class_image)

        return sample




def build_dataloader(
    instance_data_root,
    instance_prompt,
    class_data_root=None,
    class_prompt=None,
    size=512,
    batch_size=1,
    shuffle=True,
    num_workers=2):


    def collate_fn(samples):
        instance_prompts = [sample['instance_prompt'] for sample in samples]
        class_prompts = [sample['class_prompt'] for sample in samples]
        instance_images = [sample['instance_image'] for sample in samples]
        class_images = [sample['class_image'] for sample in samples]


        #instance_images = torch.cat(instance_images, dim=0)
        #if class_images[0] is not None:
        #    class_images = torch.cat(class_images, dim=0)

        samples = dict(
                instance_prompts=instance_prompts,
                class_prompts=class_prompts,
                instance_images=instance_images,
                class_images=class_images)

        return samples


    dataset = DreamBoothDataset(
            instance_data_root,
            instance_prompt,
            class_data_root,
            class_prompt,
            size)

    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn
            )

    return dataloader


if __name__=="__main__":
    instance_data_root = "dog"
    instance_prompt = "this is a V dog"

    dataloader = build_dataloader(
            instance_data_root,
            instance_prompt)

    for step, batch in enumerate(dataloader):
        import pdb; pdb.set_trace()



    dataset = DreamBoothDataset(
            instance_data_root,
            instance_prompt)

    image, prompt = dataset.__getitem__(0)
    import pdb; pdb.set_trace()





