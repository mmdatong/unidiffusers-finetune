import torch
from diffusers import UniDiffuserPipeline



def create_pipeline(model_id_or_path, device="cuda"):
    pipe = UniDiffuserPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float32)
    pipe.to(device)
    return pipe


if __name__=="__main__":
    prompt = "A photo of sks dog in a bucket"

    for ep in [99, 199, 299]:
        model_id_or_path = "outputs/{}".format(ep)
        pipe_new = create_pipeline(model_id_or_path, device="cuda")
        save_path = "dog-bucket_ep{}.png".format(ep)

        image = pipe_new(prompt, num_inference_steps=50, guidance_scale=8.0).images[0]
        image.save(save_path)



