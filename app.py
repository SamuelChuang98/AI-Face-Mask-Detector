import torch
import torchvision.transforms as transforms
import gradio as gr
from PIL import Image

classes = ["Cloth", "N95", "NoMask", "Surgical"]

model = torch.load('./models/modelAug6k.pth', weights_only=False)
model.eval()

image_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5211, 0.4858, 0.4651), (0.2889, 0.2824, 0.2880))])

def classify(image):
    image = image.convert('RGB')
    tensor = image_transforms(image).float().unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
    predicted = torch.max(output, 1)[1].item()
    if predicted == 2:
        return "This person is not wearing a mask."
    return f"This person is wearing a {classes[predicted]} mask."

with gr.Blocks(theme=gr.themes.Soft(), title="Face Mask Detector") as demo:
    gr.Markdown(
        """
        # 😷 Face Mask Detector
        ### Detect whether a person is wearing a Cloth, N95, or Surgical mask — or no mask at all.
        **Model accuracy: 79.6%** · Trained on 6,071 images across 4 classes

        ---
        *Originally a 2022 Introduction to AI group assignment — revamped in 2026 by [Samuel Chuang](https://samchuang.ca)*
        """
    )
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Image")
            submit_btn = gr.Button("Detect Mask", variant="primary")
        with gr.Column():
            result_output = gr.Textbox(label="Result", lines=3, interactive=False)
            gr.Markdown(
                """
                **Classes:**
                - 👘 Cloth mask
                - 😷 Surgical mask
                - 🛡️ N95 respirator
                - 🚫 No mask
                """
            )
    submit_btn.click(fn=classify, inputs=image_input, outputs=result_output)

demo.launch()
