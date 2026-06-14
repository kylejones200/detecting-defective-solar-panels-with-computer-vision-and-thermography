"""CLIP zero-shot prediction helpers."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from solar_panel_cv.config import ClipConfig, PipelineConfig


class ClipPredictor:
    def __init__(self, model, processor, clip_config: ClipConfig):
        self.model = model
        self.processor = processor
        self.clip_config = clip_config

    @classmethod
    def from_config(cls, config: PipelineConfig) -> ClipPredictor:
        model = CLIPModel.from_pretrained(config.model.clip_model_id)
        processor = CLIPProcessor.from_pretrained(config.model.clip_model_id)
        return cls(model, processor, config.clip)

    def _images_to_pil(self, image_batch) -> list[Image.Image]:
        return [Image.fromarray(img.numpy().astype("uint8")) for img in image_batch]

    def predict_multiclass(self, image_batch, class_names: list[str]) -> np.ndarray:
        images = self._images_to_pil(image_batch)
        text_inputs = self.processor(text=class_names, return_tensors="pt", padding=True)
        image_inputs = self.processor(images=images, return_tensors="pt", padding=True)
        with torch.no_grad():
            image_features = self.model.get_image_features(**image_inputs)
            text_features = self.model.get_text_features(**text_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        return similarity.numpy()

    def predict_binary(self, image_batch, temperature: float | None = None) -> np.ndarray:
        temperature = temperature or self.clip_config.default_temperature
        images = self._images_to_pil(image_batch)
        image_inputs = self.processor(images=images, return_tensors="pt", padding=True)
        total_predictions = np.zeros((len(images), 2))
        prompts = self.clip_config.paired_prompts
        if not prompts:
            prompts = [
                [self.clip_config.text_descriptions.clean[0], self.clip_config.text_descriptions.not_clean[0]]
            ]
        with torch.no_grad():
            image_features = self.model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            for clean_prompt, not_clean_prompt in prompts:
                text_inputs = self.processor(
                    text=[clean_prompt, not_clean_prompt],
                    return_tensors="pt",
                    padding=True,
                )
                text_features = self.model.get_text_features(**text_inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                similarity = (temperature * image_features @ text_features.T).softmax(dim=-1)
                total_predictions += similarity.numpy()
        return total_predictions / len(prompts)

    def predict_with_confidence(
        self,
        image_batch,
        class_weights: np.ndarray,
        temperature: float | None = None,
    ) -> np.ndarray:
        temperature = temperature or self.clip_config.confidence_temperature
        images = self._images_to_pil(image_batch)
        image_inputs = self.processor(images=images, return_tensors="pt", padding=True)
        clean_scores: list[torch.Tensor] = []
        not_clean_scores: list[torch.Tensor] = []
        clean_prompts = self.clip_config.text_descriptions.clean
        not_clean_prompts = self.clip_config.text_descriptions.not_clean
        with torch.no_grad():
            image_features = self.model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            for clean_prompt, not_clean_prompt in zip(clean_prompts, not_clean_prompts):
                text_inputs = self.processor(
                    text=[clean_prompt, not_clean_prompt],
                    return_tensors="pt",
                    padding=True,
                )
                text_features = self.model.get_text_features(**text_inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                similarity = temperature * image_features @ text_features.T
                clean_scores.append(similarity[:, 0])
                not_clean_scores.append(similarity[:, 1])
        clean_scores_t = torch.stack(clean_scores).mean(dim=0) * class_weights[0]
        not_clean_scores_t = torch.stack(not_clean_scores).mean(dim=0) * class_weights[1]
        scores = torch.stack([clean_scores_t, not_clean_scores_t], dim=1)
        return torch.softmax(scores, dim=1).numpy()


def evaluate_both_models(
    val_ds,
    class_names: list[str],
    mobile_model,
    clip_predictor: ClipPredictor,
) -> tuple[list[int], list[int], list[int]]:
    y_true: list[int] = []
    y_pred_mobile: list[int] = []
    y_pred_clip: list[int] = []
    for images, labels in val_ds:
        mobile_pred = mobile_model.predict(images, verbose=0)
        y_pred_mobile.extend(np.argmax(mobile_pred, axis=1))
        clip_pred = clip_predictor.predict_multiclass(images, class_names)
        y_pred_clip.extend(np.argmax(clip_pred, axis=1))
        y_true.extend(labels.numpy())
    return y_true, y_pred_mobile, y_pred_clip
