import torch
import  torchvision
import torchvision.transforms as transforms
import PIL.Image as Image

classes =["Cloth","N95","NoMask","Surgical"]

model = torch.load('./models/modelAug6k.pth', weights_only=False)
model.eval()

image_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5211, 0.4858, 0.4651), (0.2889, 0.2824, 0.2880))])

def classify(model, image_transforms, image_path, classes):
    image = Image.open(image_path).convert('RGB')
    image = image_transforms(image).float()
    image = image.unsqueeze(0)

    with torch.no_grad():
        output = model(image)

    print("Raw logits:", output)
    _, predicted = torch.max(output.data, 1)

    print("In the Image given the mask worn is a",classes[predicted.item()],"mask.")

classify(model, image_transforms, './singleImages/Indi1.jpg', classes)

img = Image.open('./singleImages/Indi1.jpg')
img.show()