import torch
import pytest
from loguru import logger
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image

image_path = "real_inputs/pixtral_transformer_inputs/people.jpg"
image = Image.open(image_path).convert("RGB")

# Your chat prompt with an image and text
input_text = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]


@pytest.mark.parametrize(
    "input_text, model_type",
    [
        (input_text, "torch_model"),
    ],
)
def test_pixtral_vlm(input_text, model_type):
    model_id = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"

    # Load processor and model
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # Convert input to string format (chat template)
    text = processor.apply_chat_template(
        input_text,
        tokenize=False,
        add_generation_prompt=True,
        padding=True,
        padding_side="left",
    )

    encoded = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        return_dict=True,
    ).to(model.device, dtype=torch.bfloat16)

    encoded = {k: v.to(dtype=torch.bfloat16) if v.dtype == torch.float32 else v for k, v in encoded.items()}

    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(
            **encoded,
            max_new_tokens=100,
            temperature=0.0,
            top_p=0.9,
            do_sample=False,
            pad_token_id=model.config.pad_token_id,
        )

    output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    logger.info(f"{model_type} output: {output}")
