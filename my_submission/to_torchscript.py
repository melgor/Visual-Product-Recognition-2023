from collections import OrderedDict

import open_clip
import torch
import torch.nn as nn
import os
import sys
import torch.nn.functional as F

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

def model_to_torchscript():
    checkpoint_path = sys.argv[1]
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Creating model and loading checkpoint')

    class Head(nn.Module):
        def __init__(self, hidden_size):
            super(Head, self).__init__()
            # self.emb = nn.Linear(hidden_size, 512, bias=False)

        def forward(self, x):
            # embeddings = self.emb(x)
            return F.normalize(x)

    class ModelToUse(nn.Module):
        def __init__(self, vit_backbone):
            super(ModelToUse, self).__init__()
            self.model = vit_backbone

            self.head = Head(768)

        def forward(self, images):
            x = self.model(images)
            return self.head(x)

    vit_backbone, model_transforms, _ = open_clip.create_model_and_transforms("ViT-L-14")
    model = ModelToUse(vit_backbone.visual)
    checkpoint = torch.load(checkpoint_path, map_location=device_type)["state_dict"]

    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        if "model.fc." in k or "model.bn." in k or "model.feature_fc." in k or "model.head" in k:
            name = k.replace("model.", "")
            new_state_dict[name] = v
        else:
            name = k.replace("model.model", "model")
            new_state_dict[name] = v

    print(model.load_state_dict(new_state_dict, strict=False))
    print('Weights are loaded')

    model.to(device).eval()
    with torch.no_grad():
        # model_scripted = torch.jit.script(model)
        model_scripted = torch.jit.trace(model, torch.randn(16, 3, 224, 224, device=device))

    folder_to_save = os.path.dirname(checkpoint_path)
    model_name = os.path.basename(checkpoint_path).split(".ckpt")[0] + ".pt"
    model_scripted.save(os.path.join(folder_to_save, model_name))
    print(f"Save: {os.path.join(folder_to_save, model_name)}")


def model_to_torchscript_convnext():
    checkpoint_path = sys.argv[1]
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Creating model and loading checkpoint')

    class Head(nn.Module):
        def __init__(self, hidden_size):
            super(Head, self).__init__()

        def forward(self, x):
            return F.normalize(x)

    class ModelToUse(nn.Module):
        def __init__(self, vit_backbone):
            super(ModelToUse, self).__init__()
            self.model = vit_backbone
            self.head = Head(1024)

        def forward(self, images):
            x = self.model(images)
            return self.head(x)

    vit_backbone, model_transforms, _ = open_clip.create_model_and_transforms("convnext_xxlarge")
    model = ModelToUse(vit_backbone.visual)
    checkpoint = torch.load(checkpoint_path, map_location=device_type)["state_dict"]

    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        k_new = k.replace("model._orig_mod.model", "model.model")
        if "model.fc." in k_new or "model.bn." in k_new or "model.feature_fc." in k_new:
            name = k_new.replace("model.", "")
            new_state_dict[name] = v
        else:
            name = k_new.replace("model.model", "model")
            new_state_dict[name] = v

    print(model.load_state_dict(new_state_dict, strict=False))
    print('Weights are loaded')

    model.to(device).eval()
    model.to(memory_format=torch.channels_last)
    with torch.no_grad():
        # model_scripted = torch.jit.script(model)
        # data = torch.randn(16, 3, 256, 256, device=device).to(memory_format=torch.channels_last).half()
        # model_scripted = torch.jit.trace(model.half(), data)
        data = torch.randn(16, 3, 256, 256, device=device).to(memory_format=torch.channels_last)
        model_scripted = torch.jit.trace(model, data)

    folder_to_save = os.path.dirname(checkpoint_path)
    model_name = os.path.basename(checkpoint_path).split(".ckpt")[0] + ".pt"
    model_scripted.save(os.path.join(folder_to_save, model_name))
    print(f"Save: {os.path.join(folder_to_save, model_name)}")

def model_to_torchscript_with_soup():
    checkpoint_paths = [
                        # "MCS2023_baseline/experiments/vit/cutout_epochepoch=02-valid_mAPvalid_mAP=0.53.ckpt",
                        # "MCS2023_baseline/experiments/vit/autoaug_epochepoch=02-valid_mAPvalid_mAP=0.53.ckpt",
                        # "MCS2023_baseline/experiments/vit/epochepoch=02-valid_mAPvalid_mAP=0.52.ckpt",
                        # "MCS2023_baseline/experiments/vit/augreg_emb512_epochepoch=03-valid_mAPvalid_mAP=0.53.ckpt"

                        'MCS2023_baseline/experiments/vit/vit_L_autoaug_p10ktrain_test/weights/epochepoch=02-valid_mAPvalid_mAP=0.56-v1.ckpt',
                        'MCS2023_baseline/experiments/vit/vit_L_autoaug_p10ktrain_test_contiue_pretrain_classifier/weights/epochepoch=02-valid_mAPvalid_mAP=0.56-v1.ckpt',
                        'MCS2023_baseline/experiments/vit/vitL-p10ktraintest_cutout_emb/weights/epochepoch=01-valid_mAPvalid_mAP=0.54.ckpt',
                        'MCS2023_baseline/experiments/vit/vitL-p10ktraintest_happy_emb/weights/epochepoch=02-valid_mAPvalid_mAP=0.55.ckpt',
                        'MCS2023_baseline/experiments/vit/vitL-p10ktraintest_randaug_emb/epochepoch=02-valid_mAPvalid_mAP=0.55.ckpt'
                        ]
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Creating model and loading checkpoint')


    class Head(nn.Module):
        def __init__(self, hidden_size):
            super(Head, self).__init__()
            # self.emb = nn.Linear(hidden_size, 512, bias=False)

        def forward(self, x):
            # embeddings = self.emb(x)
            return F.normalize(x)

    class ModelToUse(nn.Module):
        def __init__(self, vit_backbone):
            super(ModelToUse, self).__init__()
            self.model = vit_backbone

            self.head = Head(768)

        def forward(self, images):
            x = self.model(images)
            return self.head(x)

    vit_backbone, model_transforms, _ = open_clip.create_model_and_transforms("ViT-L-14")

    # Load models weights
    weight_list = []

    for path in checkpoint_paths:
        model = ModelToUse(vit_backbone.visual)
        checkpoint = torch.load(path)["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            k_new = k.replace("model._orig_mod.model", "model.model")
            if "model.fc." in k_new or "model.bn." in k_new or "model.feature_fc." in k_new or "model.head" in k_new:
                name = k_new.replace("model.", "")
                new_state_dict[name] = v
            else:
                name = k_new.replace("model.model", "model")
                new_state_dict[name] = v

        print(model.load_state_dict(new_state_dict, strict=False))
        weight_list.append(model.state_dict())

    # Average weights
    state_dict = dict((k, torch.stack([v[k] for v in weight_list]).mean(0)) for k in weight_list[0])
    model = ModelToUse(vit_backbone.visual)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    print('Weights are loaded')
    with torch.no_grad():
        # model_scripted = torch.jit.script(model)
        model_scripted = torch.jit.trace(model, torch.randn(16, 3, 224, 224, device=device))

    model_scripted.to(device).eval()

    folder_to_save = os.path.dirname(checkpoint_paths[0])
    model_name = os.path.basename(checkpoint_paths[0]).split(".ckpt")[0] + "_soup.pt"
    model_scripted.save(os.path.join(folder_to_save, model_name))
    print(f"Save: {os.path.join(folder_to_save, model_name)}")


def model_to_torchscript_convnext_soup():
    checkpoint_paths = [
        "/home/bartosz.ludwiczuk/visual-product-recognition-2023-starter-kit/experiments/convnext_p10k_amazonv2_272_fiz_2023_04_14_Apr_20_57_1681505853/weights/epochepoch=00-valid_mAPvalid_mAP=0.63.ckpt",
        "/home/bartosz.ludwiczuk/visual-product-recognition-2023-starter-kit/experiments/convnext_p10k_amazonv2_272_acc4_2023_04_15_Apr_07_14_1681542872/weights/epochepoch=00-valid_mAPvalid_mAP=0.62-v1.ckpt"
    ]
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Creating model and loading checkpoint')

    class Head(nn.Module):
        def __init__(self, hidden_size):
            super(Head, self).__init__()

        def forward(self, x):
            return F.normalize(x)

    class ModelToUse(nn.Module):
        def __init__(self, vit_backbone):
            super(ModelToUse, self).__init__()
            self.model = vit_backbone
            self.head = Head(1024)

        def forward(self, images):
            x = self.model(images)
            return self.head(x)

    vit_backbone, model_transforms, _ = open_clip.create_model_and_transforms("convnext_xxlarge")

    # Load models weights
    weight_list = []
    out = []
    # img = torch.load("image.pth").cpu()
    for path in checkpoint_paths:
        model = ModelToUse(vit_backbone.visual)
        checkpoint = torch.load(path)["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            k_new = k.replace("model._orig_mod.model", "model.model")
            if "model.fc." in k_new or "model.bn." in k_new or "model.feature_fc." in k_new:
                name = k_new.replace("model.", "")
                new_state_dict[name] = v
            else:
                name = k_new.replace("model.model", "model")
                new_state_dict[name] = v

        print(model.load_state_dict(new_state_dict, strict=False))
        model.eval()
        weight_list.append(model.state_dict())
        # with torch.no_grad():
        #     out.append(model(img))

    # Average weights
    state_dict = dict((k, torch.stack([v[k] for v in weight_list]).mean(0)) for k in weight_list[0])
    model = ModelToUse(vit_backbone.visual)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    print('Weights are loaded')
    with torch.no_grad():
        # model_scripted = torch.jit.script(model)
        model_scripted = torch.jit.trace(model, torch.randn(16, 3, 256, 256, device=device))

    model_scripted.to(device).eval()

    folder_to_save = os.path.dirname(checkpoint_paths[0])
    model_name = os.path.basename(checkpoint_paths[0]).split(".ckpt")[0] + "_soup.pt"
    model_scripted.save(os.path.join(folder_to_save, model_name))
    print(f"Save: {os.path.join(folder_to_save, model_name)}")

    del weight_list
    del model
    # with torch.no_grad():
    #     out.append(model_scripted.cpu()(img))
    # out = torch.cat(out)
    # print((out[0] - out[1]).max(), (out[0] - out[2]).max(), (out[2] - out[1]).max())


def model_to_torchscript_convnext_from_trace():
    checkpoint_path = sys.argv[1]
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Creating model and loading checkpoint')

    class Head(nn.Module):
        def __init__(self, hidden_size):
            super(Head, self).__init__()

        def forward(self, x):
            return F.normalize(x)

    class ModelToUse(nn.Module):
        def __init__(self, vit_backbone):
            super(ModelToUse, self).__init__()
            self.model = vit_backbone
            self.head = Head(1024)

        def forward(self, images):
            x = self.model(images)
            return self.head(x)

    vit_backbone, model_transforms, _ = open_clip.create_model_and_transforms("convnext_xxlarge")
    model = ModelToUse(vit_backbone.visual)
    checkpoint = torch.jit.load(checkpoint_path).state_dict()
    print(model.load_state_dict(checkpoint, strict=False))
    print('Weights are loaded')

    model.to(device).eval()
    model.to(memory_format=torch.channels_last)
    with torch.no_grad():
        # model_scripted = torch.jit.script(model)
        data = torch.randn(16, 3, 272, 272, device=device).to(memory_format=torch.channels_last).half()
        model_scripted = torch.jit.trace(model.half(), data)
        # data = torch.randn(16, 3, 256, 256, device=device).to(memory_format=torch.channels_last)
        # model_scripted = torch.jit.trace(model, data)

    folder_to_save = os.path.dirname(checkpoint_path)
    model_name = os.path.basename(checkpoint_path).split(".pt")[0] + "_fp16.pt"
    model_scripted.save(os.path.join(folder_to_save, model_name))
    print(f"Save: {os.path.join(folder_to_save, model_name)}")

# model_to_torchscript()
# model_to_torchscript_with_soup()
model_to_torchscript_convnext()
# model_to_torchscript_convnext_soup()
# model_to_torchscript_convnext_from_trace()

