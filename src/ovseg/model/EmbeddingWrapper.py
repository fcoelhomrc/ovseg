import os
import types

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm

from ovseg import OV_DATA_BASE
from ovseg.model.InferenceWrapper import preprocess
from ovseg.networks.resUNet import UNetResEncoder
from ovseg.utils.io import load_pkl


def patched_forward(self, xb):
    """
    Monkey-patches UNetResEncoder.forward to also return bottleneck and
    pre-bottleneck feature maps alongside the usual list of logits.

    Returns: (logs_list[::-1], {"bottleneck": xb, "before_bottleneck": xb_list[-1]})
    """
    xb_list = []
    logs_list = []

    for block in self.blocks_down[:-1]:
        xb = block(xb)
        xb_list.append(xb)

    # xb_list[-1] is the feature map just before the bottleneck block
    xb = self.blocks_down[-1](xb)
    # we apply global pooling to get 1 embedding per input
    embeddings = {
        "bottleneck": F.adaptive_avg_pool3d(xb, 1).flatten(1),
        "before_bottleneck": F.adaptive_avg_pool3d(xb_list[-1], 1).flatten(1),
    }

    try:
        print(
            f"Embeddings: bottleneck = {embeddings['bottleneck'].shape}, before bottleneck = {embeddings['before_bottleneck'].shape}"
        )
    except Exception as error:
        print(f"Something went wrong with embeddings: {error}, {embeddings}")

    logs_list.append(self.all_logits[-1](xb))

    for i in range(self.n_stages - 2, -1, -1):
        xb = self.upsamplings[i](xb)
        xb = torch.cat([xb, xb_list[i]], 1)
        del xb_list[i]
        xb = self.blocks_up[i](xb)
        logs_list.append(self.all_logits[i](xb))

    return logs_list[::-1], embeddings


def sliding_window(
    volume,
    network,
    dev,
    patch_size,
    sigma_gaussian_weight=1 / 8,
    overlap=0.5,
    batch_size=1,
    use_TTA=False,
    **kwargs,
):
    embeddings_dict = {
        "patch_size": patch_size,
        "coords": [],
        "embeddings": {
            "bottleneck": [],
            "before_bottleneck": [],
        },
    }

    patch_size = np.array(patch_size)

    # thanks to Fabian Isensee! I took this from his code:
    # https://github.com/MIC-DKFZ/nnUNet/blob/14992342919e63e4916c038b6dc2b050e2c62e3c/nnunet/network_architecture/neural_network.py#L250
    tmp = np.zeros(patch_size)
    center_coords = [i // 2 for i in patch_size]
    sigmas = [i * sigma_gaussian_weight for i in patch_size]
    tmp[tuple(center_coords)] = 1
    patch_weight = gaussian_filter(tmp, sigmas, 0, mode="constant", cval=0)
    patch_weight = patch_weight / np.max(patch_weight) * 1
    patch_weight = patch_weight.astype(np.float32)

    # patch_weight cannot be 0, otherwise we may end up with nans
    patch_weight[patch_weight == 0] = np.min(patch_weight[patch_weight != 0])
    patch_weight = torch.from_numpy(patch_weight).to(dev).float()

    # pad the image in case it is smaller than the patch size
    shape_in = np.array(volume.shape)
    pad = [
        0,
        patch_size[2] - shape_in[3],
        0,
        patch_size[1] - shape_in[2],
        0,
        patch_size[0] - shape_in[1],
    ]
    pad = np.maximum(pad, 0).tolist()
    volume = F.pad(volume, pad).type(torch.float)
    shape = volume.shape[1:]

    pred = torch.zeros((network.out_channels, *shape), device=dev, dtype=torch.float)
    ovlp = torch.zeros((1, *shape), device=dev, dtype=torch.float)

    nz, nx, ny = shape

    n_patches = (
        np.ceil((np.array([nz, nx, ny]) - patch_size) / (overlap * patch_size)).astype(
            int
        )
        + 1
    )

    z_list = np.linspace(0, nz - patch_size[0], n_patches[0]).astype(int).tolist()
    x_list = np.linspace(0, nx - patch_size[1], n_patches[1]).astype(int).tolist()
    y_list = np.linspace(0, ny - patch_size[2], n_patches[2]).astype(int).tolist()

    zxy_list = [(z, x, y) for z in z_list for x in x_list for y in y_list]

    n_full_batches = len(zxy_list) // batch_size
    zxy_batched = [
        zxy_list[i * batch_size : (i + 1) * batch_size] for i in range(n_full_batches)
    ]
    if n_full_batches * batch_size < len(zxy_list):
        zxy_batched.append(zxy_list[n_full_batches * batch_size :])

    with torch.no_grad():
        for zxy_batch in tqdm(zxy_batched):
            batch = torch.stack(
                [
                    volume[
                        :,
                        z : z + patch_size[0],
                        x : x + patch_size[1],
                        y : y + patch_size[2],
                    ]
                    for z, x, y in zxy_batch
                ]
            )

            if dev.startswith("cuda"):
                with torch.cuda.amp.autocast():
                    out, emb_dict = network(batch)
            else:
                out, emb_dict = network(batch)

            out = out[0]  # finest resolution logits

            for i, (z, x, y) in enumerate(zxy_batch):
                embeddings_dict["coords"].append((x, y, z))
                embeddings_dict["embeddings"]["bottleneck"].append(
                    emb_dict["bottleneck"][i].cpu().detach()
                )
                embeddings_dict["embeddings"]["before_bottleneck"].append(
                    emb_dict["before_bottleneck"][i].cpu().detach()
                )

            for i, (z, x, y) in enumerate(zxy_batch):
                pred[
                    :,
                    z : z + patch_size[0],
                    x : x + patch_size[1],
                    y : y + patch_size[2],
                ] += F.softmax(out[i], 0) * patch_weight
                ovlp[
                    :,
                    z : z + patch_size[0],
                    x : x + patch_size[1],
                    y : y + patch_size[2],
                ] += patch_weight

        pred = pred[:, : shape_in[1], : shape_in[2], : shape_in[3]]
        ovlp = ovlp[:, : shape_in[1], : shape_in[2], : shape_in[3]]

        pred[0, ovlp[0] == 0] = 1
        ovlp[ovlp == 0] = 1
        pred /= ovlp

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return pred.cpu().numpy(), embeddings_dict


def evaluate_embeddings_from_seg_model(im, spacing, model, fast=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"*** EVALUATING {model} ***")

    path_to_model = os.path.join(OV_DATA_BASE, "clara_models", model)
    model_params = load_pkl(os.path.join(path_to_model, "model_parameters.pkl"))

    orig_shape = im.shape[-3:]
    im_list = preprocess(im, spacing, model_params["preprocessing"], not fast)

    nz = np.sum([im.shape[1] for im in im_list])
    nx, ny = im_list[0].shape[2], im_list[0].shape[3]

    print("*** RUNNING THE MODEL ***")

    if model_params["architecture"] != "unetresencoder":
        raise NotImplementedError("Only implemented for ResEncoder so far...")

    # Monkey patch
    network = UNetResEncoder(**model_params["network"]).to(device)
    network.forward = types.MethodType(patched_forward, network)
    # See https://stackoverflow.com/questions/73545390/monkey-patching-class-and-instance-in-python/73545468#73545468

    n_ch = model_params["network"]["out_channels"]
    pred_params = {**model_params["prediction"], "use_TTA": not fast}

    weight_files = sorted(
        f for f in os.listdir(path_to_model) if f.startswith("network_weights")
    )
    weight_files = [os.path.join(path_to_model, f) for f in weight_files]
    if fast:
        weight_files = weight_files[:1]

    assert len(im_list) == 1, "Embedding extraction expects a single image at a time"

    pred_list = []
    embeddings_dict_per_ensemble = []

    for j, weight_file in enumerate(weight_files):
        print(f"Evaluate network {j + 1} out of {len(weight_files)}")
        network.load_state_dict(torch.load(weight_file, map_location=device))

        pred = np.zeros((n_ch, nz, nx, ny), dtype=np.float32)
        pred[:, :], embeddings_dict = sliding_window(
            im_list[0], network, device, **pred_params
        )

        pred_list.append(pred)
        embeddings_dict_per_ensemble.append(embeddings_dict)

    # convert to numpy after doing all manipulations
    bottleneck_emb_avg = torch.mean(
        torch.stack(
            [
                torch.stack(emb["embeddings"]["bottleneck"])
                for emb in embeddings_dict_per_ensemble
            ]
        ),
        dim=0,
    ).numpy()

    before_bottleneck_emb_avg = torch.mean(
        torch.stack(
            [
                torch.stack(emb["embeddings"]["before_bottleneck"])
                for emb in embeddings_dict_per_ensemble
            ]
        ),
        dim=0,
    ).numpy()

    embeddings_result = {
        "patch_size": embeddings_dict["patch_size"],
        "coords": embeddings_dict["coords"],
        "embeddings": {
            "bottleneck": bottleneck_emb_avg,
            "before_bottleneck": before_bottleneck_emb_avg,
        },
    }

    print(
        f"Bottleneck: {bottleneck_emb_avg.shape}, Before bottleneck: {before_bottleneck_emb_avg.shape}"
    )

    pred = np.stack(pred_list).mean(0)
    pred = torch.from_numpy(pred).to(device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("*** POSTPROCESSING ***")
    if "postprocessing" in model_params:
        print("WARNING: Only resizing and argmax is performed here")

    size = [int(s) for s in orig_shape]
    try:
        pred = F.interpolate(pred.unsqueeze(0), size=size, mode="trilinear")[0]
    except RuntimeError:
        print("Went out of memory. Resizing on CPU, this can be slow...")
        pred = F.interpolate(pred.unsqueeze(0).cpu(), size=size, mode="trilinear")[0]

    pred = torch.argmax(pred, 0).type(torch.float)
    pred_lb = torch.zeros_like(pred)
    for i, lb in enumerate(model_params["preprocessing"]["lb_classes"]):
        pred_lb = pred_lb + lb * (pred == i + 1).type(torch.float)

    return pred_lb, embeddings_result


def EmbeddingWrapper(im, spacing, models, fast=False):
    if not torch.cuda.is_available():
        if not fast:
            print(
                "WARNING: No GPU found, inference can be very slow, "
                'consider changing to fast mode by adding the "--fast" to '
                "your python call."
            )
        else:
            print("WARNING: No GPU found, inference can be slow.")

    if isinstance(models, str):
        models = [models]

    embeddings_diff_models = {
        "model": [],
        "embeddings": [],
    }

    pred, embeddings = evaluate_embeddings_from_seg_model(
        im, spacing, models[0], fast=fast
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    embeddings_diff_models["model"].append(models[0])
    embeddings_diff_models["embeddings"].append(embeddings)

    for model in models[1:]:
        arr, embeddings = evaluate_embeddings_from_seg_model(
            im, spacing, model, fast=fast
        )

        embeddings_diff_models["model"].append(model)
        embeddings_diff_models["embeddings"].append(embeddings)

        pred = pred * (arr == 0).type(torch.float) + arr
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pred.cpu().numpy(), embeddings_diff_models
