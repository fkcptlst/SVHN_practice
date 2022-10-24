import torch
import torch.optim as optim
from torchvision import transforms
import torch.utils.data as data
from modules.MobileNet import Model
from Dataset import Dataset

import wandb


def calculloss(digit1_logits, digit2_logits, digits_labels):
    digit1_cross_entropy = torch.nn.functional.cross_entropy(digit1_logits, digits_labels[0])
    digit2_cross_entropy = torch.nn.functional.cross_entropy(digit2_logits, digits_labels[1])
    loss = digit1_cross_entropy + digit2_cross_entropy
    return loss


def train(model, device, train_loader, optimizer, epoch):
    num_correct = 0
    for batch_idx, (images, digits_labels) in enumerate(train_loader):
        images, digits_labels = images.to(device), [digit_labels.to(device) for digit_labels in digits_labels]
        digit1_logits, digit2_logits = model.train()(images)
        loss = calculloss(digit1_logits, digit2_logits, digits_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # accuracy calculation
        digit1_prediction = digit1_logits.max(1)[1]
        digit2_prediction = digit2_logits.max(1)[1]

        num_correct += (digit1_prediction.eq(digits_labels[0]) &
                        digit2_prediction.eq(digits_labels[1])).cpu().sum()
    accuracy = 100 * num_correct.item() / len(train_loader.dataset)
    # print("train accuracy:", accuracy)
    # print("loss:", loss.item())
    return accuracy, loss.item()


def test(model, device, test_loader):
    num_correct = 0
    with torch.no_grad():
        for batch_idx, (images, digits_labels) in enumerate(test_loader):
            images, digits_labels = images.to(device), [digit_labels.to(device) for digit_labels in digits_labels]
            digit1_logits, digit2_logits = model.eval()(images)

            digit1_prediction = digit1_logits.max(1)[1]
            digit2_prediction = digit2_logits.max(1)[1]

            num_correct += (digit1_prediction.eq(digits_labels[0]) &
                            digit2_prediction.eq(digits_labels[1])).cpu().sum()

    accuracy = 100 * num_correct.item() / len(test_loader.dataset)
    # print("test accuracy:", accuracy)
    return accuracy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path_to_train_lmdb_dir = "SVHN_lmdb/train.lmdb"
path_to_val_lmdb_dir = "SVHN_lmdb/val.lmdb"

config = {
    "model_name": "mobilenet_spp_with_stn",
    "batch_size": 128,
    "optimizer": "SGD",
    "lr": 0.008,
    "momentum": 0.7,
    "mobile_net_spp_level": 3,
    "mobile_net_spp_type": "max_pool",
    "stn_spp_level": 3,
    "stn_spp_type": "max_pool",
    "stn_adaptive_pooling_shape": (54, 54)
}

wandb.init(project="svhn", config=config)

train_transform = transforms.Compose([
    # transforms.RandomRotation(degrees=(-30, 30)),
    transforms.ColorJitter(brightness=0.7, contrast=.5, saturation=.1, hue=.1),
    transforms.ToTensor(),
    transforms.RandomAffine(degrees=(-30, 30), translate=(0.2,0.2), scale=(1,1),
                            shear=60, resample=False, fill=(255,255,255)),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

train_loader = torch.utils.data.DataLoader(Dataset(path_to_train_lmdb_dir, train_transform),
                                           batch_size=config["batch_size"], shuffle=True, num_workers=0)
# test_transform = transforms.Compose([
#     transforms.RandomRotation(degrees=(-30, 30)),
#     transforms.ColorJitter(brightness=0.7, contrast=.5, saturation=.1, hue=.1),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
# ])

test_transform = train_transform

test_loader = torch.utils.data.DataLoader(Dataset(path_to_val_lmdb_dir, test_transform),
                                          batch_size=config["batch_size"], shuffle=False,
                                          num_workers=0)

model = Model(spp_level=config["mobile_net_spp_level"],
              spp_type=config["mobile_net_spp_type"],
              stn_spp_num_levels=config["stn_spp_level"],
              stn_spp_pool_type=config["stn_spp_type"],
              stn_adaptive_pooling_shape=config["stn_adaptive_pooling_shape"]).to(device)
optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])

old_test_acc = 0

# if hasattr(torch.cuda, 'empty_cache'):
# 	torch.cuda.empty_cache()

max_epochs = 50

for epoch in range(1, max_epochs + 1):
    train_accuracy, loss = train(model, device, train_loader, optimizer, epoch)
test_acc = test(model, device, test_loader)
print(f"epoch: {epoch}/{max_epochs}, train_acc: {train_accuracy}%, test_acc: {test_acc}%")
wandb.log({"train_acc": train_accuracy, "test_acc": test_acc, "loss": loss})
if test_acc >= old_test_acc:
    old_test_acc = test_acc
torch.save(model, f"trained_models/{config['model_name']}.pkl")
torch.save(model.state_dict(), f"trained_models/{config['model_name']}.pt")

wandb.save(f"trained_models/{config['model_name']}.pkl")
wandb.save(f"trained_models/{config['model_name']}.pt")

wandb.finish()
