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
    url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    # url ="https://d2luddxp0py34j.cloudfront.net/m7v8fd%2Fpreview%2F62772079%2Fmain_full.png?response-content-disposition=inline%3Bfilename%3D%22main_full.png%22%3B&response-content-type=image%2Fpng&Expires=1732847086&Signature=g4umObOD7coneLPk9KZwsWSETKWElldsSV1X6ByPPBxjzFJ847nQSaWjeXKNb~2wH6lITjkDTur97MhXk6cMUxYFBPkcSljXx6GRvZNYr9qTrB7aDdPEOOIfeuGj7wsSMNmvDepWl7b5gQrtDcTvpgIPpcGzmbIMVaglEsFFJ~ivZYCvC3mQTO1df8EYvnruJvb6N5kPEJ4M3kvkZ2Z5AoP7yX6VF7A74k57Xm3bzrDdRumUcBZp5l03gPlFgJZAszcQwNfzDPunW2-5KNQGUUo0zZYaddApeHf3EIGK~GpiyIpP4ePTsP7NSezJMN6yBWFAA0FpOMUKLMk5z6~51w__&Key-Pair-Id=APKAJT5WQLLEOADKLHBQ"
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
