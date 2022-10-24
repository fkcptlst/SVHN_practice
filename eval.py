import torch
from torchvision import transforms
from PIL import Image


def _infer(path_to_model_file, path_to_input_image):
    # model = Model()
    # model.load_state_dict(torch.load(path_to_model_file))
    # model.cuda()

    # load model
    model = torch.load(path_to_model_file)
    model.cuda()

    with torch.no_grad():
        transform = transforms.Compose([
            # transforms.Resize([64, 64]),
            # transforms.CenterCrop([54, 54]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        image = Image.open(path_to_input_image)
        image = image.convert('RGB')
        image = transform(image)
        images = image.unsqueeze(dim=0).cuda()
        print("image size:", images.size())

        digit1_logits, digit2_logits = model.eval()(images)
        # traced_script_module = torch.jit.trace(model, testimg)
        # traced_script_module.save("mobilenet.pt")

        digit1_prediction = digit1_logits.max(1)[1]
        digit2_prediction = digit2_logits.max(1)[1]

        return digit1_prediction.item(), digit2_prediction.item()
        # print(f"digit 1 distribution: {torch.nn.functional.softmax(digit1_logits, dim=1)}")
        # print(f"digit 2 distribution: {torch.nn.functional.softmax(digit2_logits, dim=1)}")

path_to_model_file= 'trained_models/mobilenet_spp.pkl'

# get filenames under testimg
import os
path_to_input_image = 'testimg'
filenames = os.listdir(path_to_input_image)
for filename in filenames:
    print(filename)
    digit1, digit2 = _infer(path_to_model_file, os.path.join(path_to_input_image, filename))
    print(f"num: {digit1}{digit2}")
    print('=====================================')