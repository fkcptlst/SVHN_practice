import torch
import torch.optim as optim
from torchvision import transforms
import torch.utils.data as data
from modules.MobileNet import Model
from Dataset import Dataset


def calculloss(digit1_logits, digit2_logits, digits_labels):
    digit1_cross_entropy = torch.nn.functional.cross_entropy(digit1_logits, digits_labels[0])
    digit2_cross_entropy = torch.nn.functional.cross_entropy(digit2_logits, digits_labels[1])
    loss = digit1_cross_entropy + digit2_cross_entropy
    return loss


def train(model, device, train_loader, optimizer, epoch):
    for batch_idx, (images, digits_labels) in enumerate(train_loader):
        images, digits_labels = images.to(device), [digit_labels.to(device) for digit_labels in digits_labels]
        digit1_logits, digit2_logits = model.train()(images)
        loss = calculloss(digit1_logits, digit2_logits, digits_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("loss:", loss.item())


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
    print("accuracy:", accuracy)
    return accuracy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path_to_train_lmdb_dir = "./SVHN/train.lmdb"
path_to_val_lmdb_dir = "./SVHN/val.lmdb"

train_transform = transforms.Compose([
    transforms.RandomRotation(degrees=(-30, 30)),
    transforms.ColorJitter(brightness=0.7, contrast=.5, saturation=.1, hue=.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
train_loader = torch.utils.data.DataLoader(Dataset(path_to_train_lmdb_dir, train_transform),
                                           batch_size=128, shuffle=True, num_workers=0)
test_transform = transforms.Compose([
    transforms.RandomRotation(degrees=(-30, 30)),
    transforms.ColorJitter(brightness=0.7, contrast=.5, saturation=.1, hue=.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
test_loader = torch.utils.data.DataLoader(Dataset(path_to_val_lmdb_dir, test_transform), batch_size=128, shuffle=False,
                                          num_workers=0)

model = Model().to(device)
optimizer = optim.SGD(model.parameters(), 0.008, momentum=0.7)

old_test_acc = 0

# if hasattr(torch.cuda, 'empty_cache'):
# 	torch.cuda.empty_cache()

max_epochs = 50

for epoch in range(1, max_epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    test_acc = test(model, device, test_loader)
    print(f"epoch: {epoch}/{max_epochs}, test_acc: {test_acc}")
    if test_acc >= old_test_acc:
        old_test_acc = test_acc
        torch.save(model, "trained_models/mobilenet.pkl")