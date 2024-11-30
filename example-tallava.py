import torch
from PIL import Image
from transformers import CLIPImageProcessor
from lmms_eval.api.instance import Instance
from lmms_eval.models.tallava import TALlava
from llava.model import load_pretrained_model
import requests

def main():
    # Load the model
    pretrained_model_path = "ToviTu/ta-llava-pretrain-phase2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    model = TALlava(pretrained=pretrained_model_path, device=device)
    print("Model loaded successfully")
    # model.eval()

    # prompts
    # url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    url = "https://www.ilankelman.org/themes2/catns.jpg"
    context = "<bos><start_of_turn>user\nDescribe the image.\n<end_of_turn>model\n"
    # context = "<bos><start_of_turn>user \n Please carefully observe the image and describe what you see.\n <end_of_turn> model \n"
    gen_kwargs = {
        "max_new_tokens": 50,
    }
    image_processor = model._image_processor

    def doc_to_visual(doc_id):
        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        image_tensor = image_processor(image, return_tensors="pt")["pixel_values"][0]
        return image_tensor

    arguments = (context, gen_kwargs, doc_to_visual, "dummy_id", "dummy_task", "dummy_split")
    metadata = {"task": "dummy_task", "doc_id": 0, "repeats": 1}

    instance = Instance(
        request_type="generate_until",
        arguments=arguments,
        idx=0,
        metadata=metadata
    )
    print("Instance created successfully:", instance)

    response = model.generate_until([instance])
    print("\nGenerated Response:", response[0])


if __name__ == "__main__":
    main()
