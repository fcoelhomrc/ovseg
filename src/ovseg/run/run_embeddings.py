import argparse
import os

from ovseg.model.EmbeddingWrapper import EmbeddingWrapper
from ovseg.utils.download_pretrained_utils import maybe_download_clara_models
from ovseg.utils.io import read_nii, save_nii


def is_nii_file(path_to_file):
    return path_to_file.endswith(".nii") or path_to_file.endswith(".nii.gz")


def run_embeddings(path_to_data, output_path, models=["pod_om"], fast=False):
    if is_nii_file(path_to_data):
        path_to_data, nii_file = os.path.split(path_to_data)
        nii_files = [nii_file]
    else:
        raise RuntimeError(f"Invalid nii file {path_to_data}")

    maybe_download_clara_models()

    pred_folder_name = "ovseg_predictions"
    for suffix in ["pod_om", "abdominal_lesions", "lymph_nodes"]:
        if suffix in models:
            pred_folder_name += f"_{suffix}"

    out_folder = os.path.join(output_path, pred_folder_name)
    os.makedirs(out_folder, exist_ok=True)

    results = {}
    for i, nii_file in enumerate(nii_files):
        print(f"Evaluate image {i} out of {len(nii_files)}")
        im, sp = read_nii(os.path.join(path_to_data, nii_file))
        pred, embeddings = EmbeddingWrapper(im, sp, models, fast=fast)

        save_nii(
            pred,
            os.path.join(out_folder, nii_file),
            os.path.join(path_to_data, nii_file),
        )

        results[nii_file] = embeddings

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path_to_data", help="Path to a single nifti file like PATH/TO/IMAGE.nii(.gz)"
    )
    parser.add_argument("output_path", help="Output folder path")
    parser.add_argument(
        "--models",
        default=["pod_om"],
        nargs="+",
        help="""Name(s) of models used during inference. Options are the following.
(i) pod_om: model for main disease sites in the pelvis/ovaries and the omentum. The two sites are encoded as 9 and 1.
(ii) abdominal_lesions: model for various lesions between the pelvis and diaphram. The model considers lesions in the omentum (1), right upper quadrant (2), left upper quadrant (3), mesenterium (5), left paracolic gutter (6) and right paracolic gutter (7).
(iii) lymph_nodes: segments disease in the lymph nodes namely infrarenal lymph nodes (13), suprarenal lymph nodes (14), supradiaphragmatic lymph nodes (15) and inguinal lymph nodes (17).
Any combination of the three are viable options.""",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        default=False,
        help="Increases inference speed by disabling dynamic z spacing, model ensembling and test-time augmentations.",
    )

    args = parser.parse_args()
    run_embeddings(args.path_to_data, args.output_path, args.models, args.fast)
