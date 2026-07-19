from __future__ import annotations

import json
import multiprocessing as mp
import queue
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Sequence

import numpy as np
import torch
import torchvision.transforms.functional as TVF
from PIL import Image
from torch import nn
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    SiglipImageProcessor,
    SiglipProcessor,
    SiglipTokenizer,
)
from transformers.utils import logging as hf_logging

hf_logging.disable_progress_bar()

try:
    from peft import PeftModel
except Exception:  # pragma: no cover - optional at import time
    PeftModel = None  # type: ignore[assignment]

from ..attention import attention_load_kwargs, attention_runtime_context, normalize_attention_backend
from ..torch_compile import generation_compile_kwargs
from ..common import (
    BatchStopFlag,
    CaptionResult,
    OUTPUTS_DIR,
    apply_torch_optimizations,
    batch_progress_line,
    clean_legacy_caption,
    coerce_image_path,
    copy_image_if_needed,
    cut_off_last_sentence,
    discover_images,
    finalize_caption_text,
    first_device,
    format_exception,
    load_rgb_image,
    log_event,
    parse_device_ids,
    remove_repeating_sentences,
    resolve_output_paths,
    reset_vram_peak_stats,
    save_caption_file,
    save_numbered_generation,
    split_round_robin,
    throttle_status,
    optimization_status_text,
    vram_usage_text,
)
from ..subprocess_runner import run_worker


CLIP_PATH = "google/siglip-so400m-patch14-384"
BASE_LLAMA = "meta-llama/Meta-Llama-3.1-8B"

ALPHA_ONE_CAPTION_TYPE_MAP = {
    ("descriptive", "formal", False, False): "Write a descriptive caption for this image in a formal tone.",
    ("descriptive", "formal", False, True): "Write a descriptive caption for this image in a formal tone within {word_count} words.",
    ("descriptive", "formal", True, False): "Write a {length} descriptive caption for this image in a formal tone.",
    ("descriptive", "informal", False, False): "Write a descriptive caption for this image in a casual tone.",
    ("descriptive", "informal", False, True): "Write a descriptive caption for this image in a casual tone within {word_count} words.",
    ("descriptive", "informal", True, False): "Write a {length} descriptive caption for this image in a casual tone.",
    ("training_prompt", "formal", False, False): "Write a stable diffusion prompt for this image.",
    ("training_prompt", "formal", False, True): "Write a stable diffusion prompt for this image within {word_count} words.",
    ("training_prompt", "formal", True, False): "Write a {length} stable diffusion prompt for this image.",
    ("rng-tags", "formal", False, False): "Write a list of Booru tags for this image.",
    ("rng-tags", "formal", False, True): "Write a list of Booru tags for this image within {word_count} words.",
    ("rng-tags", "formal", True, False): "Write a {length} list of Booru tags for this image.",
}

ALPHA_TWO_CAPTION_TYPE_MAP = {
    "Descriptive": [
        "Write a descriptive caption for this image in a formal tone.",
        "Write a descriptive caption for this image in a formal tone within {word_count} words.",
        "Write a {length} descriptive caption for this image in a formal tone.",
    ],
    "Descriptive (Informal)": [
        "Write a descriptive caption for this image in a casual tone.",
        "Write a descriptive caption for this image in a casual tone within {word_count} words.",
        "Write a {length} descriptive caption for this image in a casual tone.",
    ],
    "Training Prompt": [
        "Write a stable diffusion prompt for this image.",
        "Write a stable diffusion prompt for this image within {word_count} words.",
        "Write a {length} stable diffusion prompt for this image.",
    ],
    "MidJourney": [
        "Write a MidJourney prompt for this image.",
        "Write a MidJourney prompt for this image within {word_count} words.",
        "Write a {length} MidJourney prompt for this image.",
    ],
    "Booru tag list": [
        "Write a list of Booru tags for this image.",
        "Write a list of Booru tags for this image within {word_count} words.",
        "Write a {length} list of Booru tags for this image.",
    ],
    "Booru-like tag list": [
        "Write a list of Booru-like tags for this image.",
        "Write a list of Booru-like tags for this image within {word_count} words.",
        "Write a {length} list of Booru-like tags for this image.",
    ],
    "Art Critic": [
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.",
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it within {word_count} words.",
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it {length}.",
    ],
    "Product Listing": [
        "Write a caption for this image as though it were a product listing.",
        "Write a caption for this image as though it were a product listing. Keep it under {word_count} words.",
        "Write a {length} caption for this image as though it were a product listing.",
    ],
    "Social Media Post": [
        "Write a caption for this image as if it were being used for a social media post.",
        "Write a caption for this image as if it were being used for a social media post. Limit the caption to {word_count} words.",
        "Write a {length} caption for this image as if it were being used for a social media post.",
    ],
}


class PreAlphaImageAdapter(nn.Module):
    def __init__(self, input_features: int, output_features: int):
        super().__init__()
        self.linear1 = nn.Linear(input_features, output_features)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(output_features, output_features)

    def forward(self, vision_outputs: torch.Tensor) -> torch.Tensor:
        x = self.linear1(vision_outputs)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class ChatImageAdapter(nn.Module):
    def __init__(
        self,
        input_features: int,
        output_features: int,
        ln1: bool = False,
        pos_emb: bool = False,
        num_image_tokens: int = 38,
        deep_extract: bool = False,
    ):
        super().__init__()
        self.deep_extract = deep_extract
        if self.deep_extract:
            input_features = input_features * 5
        self.linear1 = nn.Linear(input_features, output_features)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(output_features, output_features)
        self.ln1 = nn.Identity() if not ln1 else nn.LayerNorm(input_features)
        self.pos_emb = None if not pos_emb else nn.Parameter(torch.zeros(num_image_tokens, input_features))
        self.other_tokens = nn.Embedding(3, output_features)
        self.other_tokens.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, vision_outputs: Sequence[torch.Tensor]) -> torch.Tensor:
        if self.deep_extract:
            x = torch.concat(
                (
                    vision_outputs[-2],
                    vision_outputs[3],
                    vision_outputs[7],
                    vision_outputs[13],
                    vision_outputs[20],
                ),
                dim=-1,
            )
        else:
            x = vision_outputs[-2]
        x = self.ln1(x)
        if self.pos_emb is not None:
            x = x + self.pos_emb
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        other_tokens = self.other_tokens(torch.tensor([0, 1], device=self.other_tokens.weight.device).expand(x.shape[0], -1))
        return torch.cat((other_tokens[:, 0:1], x, other_tokens[:, 1:2]), dim=1)

    def get_eot_embedding(self) -> torch.Tensor:
        return self.other_tokens(torch.tensor([2], device=self.other_tokens.weight.device)).squeeze(0)


@dataclass(frozen=True)
class LegacyVariantConfig:
    key: str
    title: str
    checkpoint_dir: Path
    mode: str
    base_model_name: str = BASE_LLAMA
    clip_path: str = CLIP_PATH
    default_dtype: torch.dtype = torch.bfloat16
    default_prompt: str = "A descriptive caption for this image:\n"
    clean_aggressive: bool = False


@dataclass
class LegacyModelBundle:
    device: str
    text_model: Any
    image_adapter: nn.Module
    clip_model: nn.Module
    tokenizer: Any
    processor: Any


@dataclass
class LegacyGenerationStats:
    generated_tokens: int = 0
    elapsed_seconds: float = 0.0
    tokens_per_second: float = 0.0


def _generation_stats_text(stats: LegacyGenerationStats) -> str:
    return (
        f"{stats.generated_tokens} token(s) in {stats.elapsed_seconds:.2f}s "
        f"({stats.tokens_per_second:.2f} tok/s)"
    )


def _quant_config(use_4bit: bool, dtype: torch.dtype) -> BitsAndBytesConfig | None:
    if not use_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def _resolve_device(device_id: int | str) -> str:
    if str(device_id).lower() == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Use device 'cpu' only if the selected model fits system RAM.")
    return f"cuda:{int(device_id)}"


def _stats_device_key(device_id: int | str) -> str:
    text = str(device_id).lower()
    if text == "cpu" or text.startswith("cuda"):
        return text
    return f"cuda:{int(device_id)}"


def _synchronize_if_cuda(device: Any) -> None:
    if not torch.cuda.is_available():
        return
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


@contextmanager
def _quiet_known_siglip_config_warning():
    previous = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    try:
        yield
    finally:
        hf_logging.set_verbosity(previous)


def _token_id(value: Any) -> int | None:
    if value is None:
        return None
    try:
        token_id = int(value)
    except (TypeError, ValueError):
        return None
    return token_id if token_id >= 0 else None


def _pad_token_id(tokenizer: Any) -> int | None:
    return _token_id(getattr(tokenizer, "pad_token_id", None)) or _token_id(getattr(tokenizer, "eos_token_id", None))


def _pad_generation_kwargs(tokenizer: Any) -> dict[str, int]:
    pad_id = _pad_token_id(tokenizer)
    return {"pad_token_id": pad_id} if pad_id is not None else {}


def _normalize_generation_config(model: Any, tokenizer: Any) -> None:
    generation_config = getattr(model, "generation_config", None)
    if generation_config is None:
        return
    pad_id = _pad_token_id(tokenizer)
    if pad_id is not None:
        generation_config.pad_token_id = pad_id
    if hasattr(generation_config, "max_length"):
        generation_config.max_length = None


def _hidden_size(model: Any) -> int:
    candidates = [
        getattr(model, "config", None),
        getattr(getattr(model, "base_model", None), "config", None),
        getattr(getattr(getattr(model, "base_model", None), "model", None), "config", None),
        getattr(getattr(getattr(getattr(model, "base_model", None), "model", None), "model", None), "config", None),
    ]
    for cfg in candidates:
        value = getattr(cfg, "hidden_size", None)
        if value:
            return int(value)
    raise RuntimeError("Could not determine language model hidden size.")


def _embed_tokens(model: Any):
    paths = [
        ("model", "embed_tokens"),
        ("model", "model", "embed_tokens"),
        ("base_model", "model", "model", "embed_tokens"),
        ("base_model", "model", "model", "model", "embed_tokens"),
        ("language_model", "model", "embed_tokens"),
    ]
    for path in paths:
        current = model
        for attr in path:
            current = getattr(current, attr, None)
            if current is None:
                break
        if current is not None:
            return current
    raise RuntimeError("Could not locate token embedding layer on the text model.")


def _decode_generated_rows(
    tokenizer: Any,
    token_rows: torch.Tensor,
    *,
    skip_special_tokens: bool,
    strip_token_ids: Sequence[int | None] = (),
    strip_last: bool = False,
) -> list[str]:
    strip_ids = {int(token_id) for token_id in strip_token_ids if token_id is not None and int(token_id) >= 0}
    captions: list[str] = []
    for row in token_rows:
        tokens = row
        if tokens.shape[0] > 0 and strip_last:
            tokens = tokens[:-1]
        elif tokens.shape[0] > 0 and strip_ids and int(tokens[-1].item()) in strip_ids:
            tokens = tokens[:-1]
        captions.append(tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=False))
    return captions


class LegacySiglipEngine:
    def __init__(self, config: LegacyVariantConfig):
        self.config = config
        self.stop_flag = BatchStopFlag()
        self._bundle_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._bundles: dict[tuple[str, bool, bool], LegacyModelBundle] = {}
        self._tokenizer_source_cache: str | Path | None = None
        self.last_generation_stats = LegacyGenerationStats()
        self._last_generation_stats_by_device: dict[str, LegacyGenerationStats] = {}

    def _record_generation_stats(self, device: int | str, generated_tokens: int, elapsed_seconds: float) -> LegacyGenerationStats:
        elapsed = max(float(elapsed_seconds), 0.0)
        stats = LegacyGenerationStats(
            generated_tokens=max(0, int(generated_tokens)),
            elapsed_seconds=elapsed,
            tokens_per_second=max(0, int(generated_tokens)) / max(elapsed, 1e-9),
        )
        with self._stats_lock:
            self.last_generation_stats = stats
            self._last_generation_stats_by_device[_stats_device_key(device)] = stats
        return stats

    def _generation_stats_for_device(self, device: int | str) -> LegacyGenerationStats:
        with self._stats_lock:
            return self._last_generation_stats_by_device.get(_stats_device_key(device), self.last_generation_stats)

    def _should_save_image(self, settings: dict[str, Any]) -> bool:
        if "save_image" in settings:
            return bool(settings.get("save_image"))
        return not bool(settings.get("dont_save_image", False))

    def stop_batch(self) -> str:
        self.stop_flag.stop()
        return "Stopping batch processing after the current image finishes."

    def _tokenizer_source(self) -> str | Path:
        if self._tokenizer_source_cache is not None:
            return self._tokenizer_source_cache
        text_dir = self.config.checkpoint_dir / "text_model"
        if (text_dir / "tokenizer_config.json").exists() or (text_dir / "tokenizer.json").exists():
            self._tokenizer_source_cache = text_dir
        elif (text_dir / "adapter_config.json").exists():
            adapter_data = json.loads((text_dir / "adapter_config.json").read_text(encoding="utf-8"))
            self._tokenizer_source_cache = adapter_data.get("base_model_name_or_path") or self.config.base_model_name
        else:
            self._tokenizer_source_cache = self.config.base_model_name
        return self._tokenizer_source_cache

    def _load_text_model(
        self,
        device: str,
        dtype: torch.dtype,
        use_4bit: bool,
        low_cpu_mem_usage: bool = False,
        attention_backend: str = "sdpa",
    ) -> Any:
        text_dir = self.config.checkpoint_dir / "text_model"
        log_event(f"Loading text model for {self.config.title}: source={text_dir}", self.config.title)
        quantization_config = _quant_config(use_4bit, dtype)
        kwargs: dict[str, Any] = {
            "device_map": device,
            "dtype": dtype,
        }
        if low_cpu_mem_usage:
            kwargs["low_cpu_mem_usage"] = True
        quant_name = "nf4" if use_4bit else ("fp16" if dtype == torch.float16 else "bf16")
        kwargs.update(attention_load_kwargs({"attention_backend": attention_backend}, quant=quant_name))
        if quantization_config is not None:
            kwargs["quantization_config"] = quantization_config

        adapter_config = text_dir / "adapter_config.json"
        if adapter_config.exists():
            if PeftModel is None:
                raise RuntimeError("PEFT is required to load the local LoRA adapter but is not installed.")
            adapter_data = json.loads(adapter_config.read_text(encoding="utf-8"))
            base_model_name = adapter_data.get("base_model_name_or_path") or self.config.base_model_name
            base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **kwargs)
            model = PeftModel.from_pretrained(base_model, text_dir)
        elif text_dir.exists() and any(text_dir.iterdir()):
            model = AutoModelForCausalLM.from_pretrained(text_dir, local_files_only=True, **kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(self.config.base_model_name, **kwargs)
        model.eval()
        log_event(f"Text model ready on {device}.", self.config.title)
        return model

    def _load_clip_model(self, device: str) -> nn.Module:
        log_event(f"Loading SigLIP vision model: {self.config.clip_path}", self.config.title)
        with _quiet_known_siglip_config_warning():
            clip_base = AutoModel.from_pretrained(self.config.clip_path)
        clip_model = clip_base.vision_model
        clip_weights = self.config.checkpoint_dir / "clip_model.pt"
        if clip_weights.exists():
            checkpoint = torch.load(clip_weights, map_location="cpu")
            checkpoint = {str(k).replace("_orig_mod.module.", ""): v for k, v in checkpoint.items()}
            clip_model.load_state_dict(checkpoint)
        clip_model.eval()
        clip_model.requires_grad_(False)
        clip_model = clip_model.to(device)
        log_event(f"SigLIP vision model ready on {device}.", self.config.title)
        return clip_model

    def _load_image_adapter(self, clip_model: nn.Module, text_model: Any, device: str) -> nn.Module:
        log_event("Loading image adapter.", self.config.title)
        text_hidden = _hidden_size(text_model)
        if self.config.mode == "pre_alpha":
            adapter: nn.Module = PreAlphaImageAdapter(clip_model.config.hidden_size, text_hidden)
        else:
            adapter = ChatImageAdapter(clip_model.config.hidden_size, text_hidden, False, False, 38, False)
        adapter_path = self.config.checkpoint_dir / "image_adapter.pt"
        if not adapter_path.exists():
            raise FileNotFoundError(f"Missing image adapter: {adapter_path}")
        state = torch.load(adapter_path, map_location="cpu")
        adapter.load_state_dict(state)
        adapter.eval()
        adapter = adapter.to(device)
        log_event("Image adapter ready.", self.config.title)
        return adapter

    def _bundle(
        self,
        device_id: int | str,
        use_fp16: bool = False,
        use_4bit: bool = False,
        low_cpu_mem_usage: bool = False,
        attention_backend: str = "sdpa",
    ) -> LegacyModelBundle:
        device = _resolve_device(device_id)
        backend = normalize_attention_backend(attention_backend)
        key = (device, bool(use_fp16), bool(use_4bit), bool(low_cpu_mem_usage), backend)
        with self._bundle_lock:
            if key in self._bundles:
                log_event(f"Using cached model bundle on {device}.", self.config.title)
                return self._bundles[key]
            log_event(
                f"Preparing model bundle on {device} | dtype={'fp16' if use_fp16 else str(self.config.default_dtype)} | 4bit={bool(use_4bit)}",
                self.config.title,
            )
            if device.startswith("cuda"):
                torch.cuda.set_device(int(device.split(":")[1]))
            dtype = torch.float16 if use_fp16 else self.config.default_dtype
            log_event("Loading SigLIP processor.", self.config.title)
            processor = SiglipProcessor(
                image_processor=SiglipImageProcessor.from_pretrained(self.config.clip_path),
                tokenizer=SiglipTokenizer.from_pretrained(self.config.clip_path),
            )
            log_event("Loading caption tokenizer.", self.config.title)
            tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_source(), use_fast=False)
            if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
                tokenizer.pad_token = tokenizer.eos_token
            if hasattr(tokenizer, "padding_side"):
                tokenizer.padding_side = "left"
            text_model = self._load_text_model(device, dtype, use_4bit, low_cpu_mem_usage, backend)
            _normalize_generation_config(text_model, tokenizer)
            clip_model = self._load_clip_model(device)
            image_adapter = self._load_image_adapter(clip_model, text_model, device)
            bundle = LegacyModelBundle(
                device=device,
                text_model=text_model,
                image_adapter=image_adapter,
                clip_model=clip_model,
                tokenizer=tokenizer,
                processor=processor,
            )
            self._bundles[key] = bundle
            log_event("Model bundle ready.", self.config.title)
            return bundle

    def build_alpha_one_prompt(self, caption_type: str, caption_tone: str, caption_length: str | int, custom_prompt: str) -> str:
        if custom_prompt and custom_prompt.strip():
            return custom_prompt.strip()
        length: str | int | None = None if caption_length == "any" else caption_length
        if isinstance(length, str):
            try:
                length = int(length)
            except ValueError:
                pass
        if caption_type in {"rng-tags", "training_prompt"}:
            caption_tone = "formal"
        key = (caption_type, caption_tone, isinstance(length, str), isinstance(length, int))
        template = ALPHA_ONE_CAPTION_TYPE_MAP.get(key)
        if not template:
            raise ValueError(f"Invalid Alpha One caption options: {key}")
        return template.format(length=caption_length, word_count=caption_length)

    def build_alpha_two_prompt(
        self,
        caption_type: str,
        caption_length: str | int,
        extra_options: Sequence[str] | None,
        name_input: str,
        custom_prompt: str,
    ) -> str:
        if custom_prompt and custom_prompt.strip():
            return custom_prompt.strip()
        if caption_type not in ALPHA_TWO_CAPTION_TYPE_MAP:
            raise ValueError(f"Invalid Alpha Two caption type: {caption_type}")
        if caption_length == "any":
            idx = 0
        elif isinstance(caption_length, str) and caption_length.isdigit():
            idx = 1
        else:
            idx = 2
        prompt = ALPHA_TWO_CAPTION_TYPE_MAP[caption_type][idx]
        selected = [opt for opt in (extra_options or []) if opt]
        if selected:
            prompt += " " + " ".join(selected)
        return (
            prompt.replace("{name}", name_input or "{NAME}")
            .replace("{length}", str(caption_length))
            .replace("{word_count}", str(caption_length))
        ).strip()

    @torch.inference_mode()
    def _generate_pre_alpha(self, image: Image.Image, settings: dict[str, Any], bundle: LegacyModelBundle) -> tuple[str, str]:
        log_event("Pre-Alpha: preprocessing image and prompt.", self.config.title)
        device = bundle.device
        prompt_text = (settings.get("custom_prompt") or self.config.default_prompt).strip() or self.config.default_prompt
        pixel_values = bundle.processor(images=image, return_tensors="pt").pixel_values.to(device)
        prompt = bundle.tokenizer.encode(
            prompt_text,
            return_tensors="pt",
            padding=bool(settings.get("padding", False)),
            truncation=bool(settings.get("truncation", False)),
            add_special_tokens=bool(settings.get("add_special_tokens", False)),
        ).to(device)
        autocast_enabled = device.startswith("cuda")
        with torch.amp.autocast(device_type="cuda", enabled=autocast_enabled):
            vision_outputs = bundle.clip_model(pixel_values=pixel_values, output_hidden_states=True)
            image_features = vision_outputs.hidden_states[-2]
            embedded_images = bundle.image_adapter(image_features)
        embeds = _embed_tokens(bundle.text_model)
        prompt_embeds = embeds(prompt)
        bos = torch.tensor([[bundle.tokenizer.bos_token_id]], device=device, dtype=torch.int64)
        embedded_bos = embeds(bos)
        inputs_embeds = torch.cat(
            [
                embedded_bos.expand(embedded_images.shape[0], -1, -1),
                embedded_images.to(dtype=embedded_bos.dtype),
                prompt_embeds.expand(embedded_images.shape[0], -1, -1),
            ],
            dim=1,
        )
        input_ids = torch.cat(
            [
                bos,
                torch.zeros((1, embedded_images.shape[1]), dtype=torch.long, device=device),
                prompt,
            ],
            dim=1,
        )
        do_sample = bool(settings.get("do_sample", False))
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": int(settings.get("max_new_tokens", 300)),
            "do_sample": do_sample,
            "suppress_tokens": None,
            **_pad_generation_kwargs(bundle.tokenizer),
        }
        if do_sample:
            generation_kwargs["top_k"] = int(settings.get("top_k", 10))
            generation_kwargs["temperature"] = max(float(settings.get("temperature", 0.5)), 1e-5)
        log_event(f"Pre-Alpha: generating caption (max_new_tokens={generation_kwargs['max_new_tokens']}).", self.config.title)
        self._record_generation_stats(device, 0, 0.0)
        _synchronize_if_cuda(device)
        generation_started = time.time()
        with attention_runtime_context(settings):
            generate_ids = bundle.text_model.generate(
                input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=torch.ones_like(input_ids),
                **generation_kwargs,
                **generation_compile_kwargs(settings, bundle.text_model),
            )
        _synchronize_if_cuda(device)
        generation_elapsed = max(time.time() - generation_started, 1e-9)
        generate_ids = generate_ids[:, input_ids.shape[1] :]
        if generate_ids.numel() and generate_ids[0][-1] == bundle.tokenizer.eos_token_id:
            generate_ids = generate_ids[:, :-1]
        generated_token_count = max(0, int(generate_ids.shape[-1]))
        stats = self._record_generation_stats(device, generated_token_count, generation_elapsed)
        caption = bundle.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        log_event(f"Pre-Alpha: generation complete. Token speed: {_generation_stats_text(stats)}.", self.config.title)
        return prompt_text, self._postprocess(caption, settings)

    @torch.inference_mode()
    def _generate_alpha_one(self, image: Image.Image, settings: dict[str, Any], bundle: LegacyModelBundle) -> tuple[str, str]:
        log_event("Alpha 1: preprocessing image and prompt.", self.config.title)
        prompt_text = self.build_alpha_one_prompt(
            str(settings.get("caption_type", "descriptive")),
            str(settings.get("caption_tone", "formal")),
            settings.get("caption_length", "any"),
            str(settings.get("custom_prompt", "")),
        )
        device = bundle.device
        pixel_values = bundle.processor(images=image, return_tensors="pt").pixel_values.to(device)
        prompt = bundle.tokenizer.encode(prompt_text, return_tensors="pt", padding=False, truncation=False, add_special_tokens=False).to(device)
        with torch.amp.autocast(device_type="cuda", enabled=device.startswith("cuda")):
            vision_outputs = bundle.clip_model(pixel_values=pixel_values, output_hidden_states=True)
            embedded_images = bundle.image_adapter(vision_outputs.hidden_states)
        embeds = _embed_tokens(bundle.text_model)
        prompt_embeds = embeds(prompt)
        bos = torch.tensor([[bundle.tokenizer.bos_token_id]], device=device, dtype=torch.int64)
        embedded_bos = embeds(bos)
        eot_id = bundle.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if not isinstance(eot_id, int) or eot_id < 0:
            eot_id = bundle.tokenizer.eos_token_id
        eot_embed = bundle.image_adapter.get_eot_embedding().unsqueeze(0).to(dtype=embedded_bos.dtype, device=device)
        inputs_embeds = torch.cat(
            [
                embedded_bos.expand(embedded_images.shape[0], -1, -1),
                embedded_images.to(dtype=embedded_bos.dtype),
                prompt_embeds.expand(embedded_images.shape[0], -1, -1),
                eot_embed.expand(embedded_images.shape[0], -1, -1),
            ],
            dim=1,
        )
        input_ids = torch.cat(
            [
                bos,
                torch.zeros((1, embedded_images.shape[1]), dtype=torch.long, device=device),
                prompt,
                torch.tensor([[eot_id]], dtype=torch.long, device=device),
            ],
            dim=1,
        )
        log_event(f"Alpha 1: generating caption (max_new_tokens={int(settings.get('max_new_tokens', 300))}).", self.config.title)
        self._record_generation_stats(device, 0, 0.0)
        _synchronize_if_cuda(device)
        generation_started = time.time()
        with attention_runtime_context(settings):
            generate_ids = bundle.text_model.generate(
                input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=int(settings.get("max_new_tokens", 300)),
                do_sample=True,
                suppress_tokens=None,
                **_pad_generation_kwargs(bundle.tokenizer),
                **generation_compile_kwargs(settings, bundle.text_model),
            )
        _synchronize_if_cuda(device)
        generation_elapsed = max(time.time() - generation_started, 1e-9)
        generate_ids = generate_ids[:, input_ids.shape[1] :]
        if generate_ids.shape[1] > 0:
            generate_ids = generate_ids[:, :-1]
        generated_token_count = max(0, int(generate_ids.shape[-1]))
        stats = self._record_generation_stats(device, generated_token_count, generation_elapsed)
        caption = bundle.tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
        log_event(f"Alpha 1: generation complete. Token speed: {_generation_stats_text(stats)}.", self.config.title)
        return prompt_text, self._postprocess(caption, settings)

    @torch.inference_mode()
    def _generate_alpha_two(self, image: Image.Image, settings: dict[str, Any], bundle: LegacyModelBundle) -> tuple[str, str]:
        log_event("Alpha 2: preprocessing image and prompt.", self.config.title)
        prompt_text = self.build_alpha_two_prompt(
            str(settings.get("caption_type", "Descriptive")),
            settings.get("caption_length", "long"),
            settings.get("extra_options") or [],
            str(settings.get("name_input", "")),
            str(settings.get("custom_prompt", "")),
        )
        device = bundle.device
        resized = image.resize((384, 384), Image.Resampling.LANCZOS)
        pixel_values = TVF.pil_to_tensor(resized).unsqueeze(0) / 255.0
        pixel_values = TVF.normalize(pixel_values, [0.5], [0.5]).to(device)
        with torch.amp.autocast(device_type="cuda", enabled=device.startswith("cuda")):
            vision_outputs = bundle.clip_model(pixel_values=pixel_values, output_hidden_states=True)
            embedded_images = bundle.image_adapter(vision_outputs.hidden_states)
        convo = [
            {"role": "system", "content": "You are a helpful image captioner."},
            {"role": "user", "content": prompt_text},
        ]
        if hasattr(bundle.tokenizer, "apply_chat_template"):
            convo_string = bundle.tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
        else:
            convo_string = prompt_text
        convo_tokens = bundle.tokenizer.encode(convo_string, return_tensors="pt", add_special_tokens=False, truncation=False).squeeze(0).to(device)
        prompt_tokens = bundle.tokenizer.encode(prompt_text, return_tensors="pt", add_special_tokens=False, truncation=False).squeeze(0).to(device)
        eot_id = bundle.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        eot_indices = (convo_tokens == eot_id).nonzero(as_tuple=True)[0].tolist() if isinstance(eot_id, int) else []
        preamble_len = eot_indices[1] - prompt_tokens.shape[0] if len(eot_indices) >= 2 else max(convo_tokens.shape[0] - prompt_tokens.shape[0], 0)
        embeds = _embed_tokens(bundle.text_model)
        convo_embeds = embeds(convo_tokens.unsqueeze(0))
        input_embeds = torch.cat(
            [
                convo_embeds[:, :preamble_len],
                embedded_images.to(dtype=convo_embeds.dtype),
                convo_embeds[:, preamble_len:],
            ],
            dim=1,
        ).to(device)
        input_ids = torch.cat(
            [
                convo_tokens[:preamble_len].unsqueeze(0),
                torch.zeros((1, embedded_images.shape[1]), dtype=torch.long, device=device),
                convo_tokens[preamble_len:].unsqueeze(0),
            ],
            dim=1,
        )
        log_event(f"Alpha 2: generating caption (max_new_tokens={int(settings.get('max_new_tokens', 300))}).", self.config.title)
        self._record_generation_stats(device, 0, 0.0)
        _synchronize_if_cuda(device)
        generation_started = time.time()
        with attention_runtime_context(settings):
            generate_ids = bundle.text_model.generate(
                input_ids,
                inputs_embeds=input_embeds,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=int(settings.get("max_new_tokens", 300)),
                do_sample=True,
                suppress_tokens=None,
                **_pad_generation_kwargs(bundle.tokenizer),
                **generation_compile_kwargs(settings, bundle.text_model),
            )
        _synchronize_if_cuda(device)
        generation_elapsed = max(time.time() - generation_started, 1e-9)
        generate_ids = generate_ids[:, input_ids.shape[1] :]
        if generate_ids.shape[1] > 0:
            last = generate_ids[0][-1]
            if last == bundle.tokenizer.eos_token_id or (isinstance(eot_id, int) and last == eot_id):
                generate_ids = generate_ids[:, :-1]
        generated_token_count = max(0, int(generate_ids.shape[-1]))
        stats = self._record_generation_stats(device, generated_token_count, generation_elapsed)
        caption = bundle.tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
        log_event(f"Alpha 2: generation complete. Token speed: {_generation_stats_text(stats)}.", self.config.title)
        return prompt_text, self._postprocess(caption, settings)

    @torch.inference_mode()
    def _generate_pre_alpha_batch(self, images: Sequence[Image.Image], settings: dict[str, Any], bundle: LegacyModelBundle) -> tuple[str, list[str]]:
        log_event(f"Pre-Alpha: preprocessing true batch of {len(images)} image(s).", self.config.title)
        device = bundle.device
        prompt_text = (settings.get("custom_prompt") or self.config.default_prompt).strip() or self.config.default_prompt
        pixel_values = bundle.processor(images=list(images), return_tensors="pt").pixel_values.to(device)
        batch_count = int(pixel_values.shape[0])
        prompt = bundle.tokenizer.encode(
            prompt_text,
            return_tensors="pt",
            padding=bool(settings.get("padding", False)),
            truncation=bool(settings.get("truncation", False)),
            add_special_tokens=bool(settings.get("add_special_tokens", False)),
        ).to(device)
        autocast_enabled = device.startswith("cuda")
        with torch.amp.autocast(device_type="cuda", enabled=autocast_enabled):
            vision_outputs = bundle.clip_model(pixel_values=pixel_values, output_hidden_states=True)
            image_features = vision_outputs.hidden_states[-2]
            embedded_images = bundle.image_adapter(image_features)
        embeds = _embed_tokens(bundle.text_model)
        prompt_embeds = embeds(prompt)
        bos = torch.full((batch_count, 1), int(bundle.tokenizer.bos_token_id), device=device, dtype=torch.int64)
        embedded_bos = embeds(bos)
        inputs_embeds = torch.cat(
            [
                embedded_bos,
                embedded_images.to(dtype=embedded_bos.dtype),
                prompt_embeds.expand(batch_count, -1, -1),
            ],
            dim=1,
        )
        input_ids = torch.cat(
            [
                bos,
                torch.zeros((batch_count, embedded_images.shape[1]), dtype=torch.long, device=device),
                prompt.expand(batch_count, -1),
            ],
            dim=1,
        )
        do_sample = bool(settings.get("do_sample", False))
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": int(settings.get("max_new_tokens", 300)),
            "do_sample": do_sample,
            "suppress_tokens": None,
            **_pad_generation_kwargs(bundle.tokenizer),
        }
        if do_sample:
            generation_kwargs["top_k"] = int(settings.get("top_k", 10))
            generation_kwargs["temperature"] = max(float(settings.get("temperature", 0.5)), 1e-5)
        log_event(
            f"Pre-Alpha: generating {batch_count} caption(s) in one batch (max_new_tokens={generation_kwargs['max_new_tokens']}).",
            self.config.title,
        )
        self._record_generation_stats(device, 0, 0.0)
        _synchronize_if_cuda(device)
        generation_started = time.time()
        with attention_runtime_context(settings):
            generate_ids = bundle.text_model.generate(
                input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=torch.ones_like(input_ids),
                **generation_kwargs,
                **generation_compile_kwargs(settings, bundle.text_model),
            )
        _synchronize_if_cuda(device)
        generation_elapsed = max(time.time() - generation_started, 1e-9)
        generated_rows = generate_ids[:, input_ids.shape[1] :]
        generated_token_count = max(0, int(generated_rows.numel()))
        stats = self._record_generation_stats(device, generated_token_count, generation_elapsed)
        captions = _decode_generated_rows(
            bundle.tokenizer,
            generated_rows,
            skip_special_tokens=True,
            strip_token_ids=[bundle.tokenizer.eos_token_id],
        )
        log_event(f"Pre-Alpha: batch generation complete. Token speed: {_generation_stats_text(stats)}.", self.config.title)
        return prompt_text, [self._postprocess(caption, settings) for caption in captions]

    @torch.inference_mode()
    def _generate_alpha_one_batch(self, images: Sequence[Image.Image], settings: dict[str, Any], bundle: LegacyModelBundle) -> tuple[str, list[str]]:
        prompt_text = self.build_alpha_one_prompt(
            str(settings.get("caption_type", "descriptive")),
            str(settings.get("caption_tone", "formal")),
            settings.get("caption_length", "any"),
            str(settings.get("custom_prompt", "")),
        )
        device = bundle.device
        pixel_values = bundle.processor(images=list(images), return_tensors="pt").pixel_values.to(device)
        batch_count = int(pixel_values.shape[0])
        prompt = bundle.tokenizer.encode(prompt_text, return_tensors="pt", padding=False, truncation=False, add_special_tokens=False).to(device)
        with torch.amp.autocast(device_type="cuda", enabled=device.startswith("cuda")):
            vision_outputs = bundle.clip_model(pixel_values=pixel_values, output_hidden_states=True)
            embedded_images = bundle.image_adapter(vision_outputs.hidden_states)
        embeds = _embed_tokens(bundle.text_model)
        prompt_embeds = embeds(prompt)
        bos = torch.full((batch_count, 1), int(bundle.tokenizer.bos_token_id), device=device, dtype=torch.int64)
        embedded_bos = embeds(bos)
        eot_id = bundle.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if not isinstance(eot_id, int) or eot_id < 0:
            eot_id = bundle.tokenizer.eos_token_id
        eot_embed = bundle.image_adapter.get_eot_embedding().unsqueeze(0).to(dtype=embedded_bos.dtype, device=device)
        eot_tokens = torch.full((batch_count, 1), int(eot_id), dtype=torch.long, device=device)
        inputs_embeds = torch.cat(
            [
                embedded_bos,
                embedded_images.to(dtype=embedded_bos.dtype),
                prompt_embeds.expand(batch_count, -1, -1),
                eot_embed.expand(batch_count, -1, -1),
            ],
            dim=1,
        )
        input_ids = torch.cat(
            [
                bos,
                torch.zeros((batch_count, embedded_images.shape[1]), dtype=torch.long, device=device),
                prompt.expand(batch_count, -1),
                eot_tokens,
            ],
            dim=1,
        )
        log_event(
            f"Alpha 1: generating {batch_count} caption(s) in one batch (max_new_tokens={int(settings.get('max_new_tokens', 300))}).",
            self.config.title,
        )
        self._record_generation_stats(device, 0, 0.0)
        _synchronize_if_cuda(device)
        generation_started = time.time()
        with attention_runtime_context(settings):
            generate_ids = bundle.text_model.generate(
                input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=int(settings.get("max_new_tokens", 300)),
                do_sample=True,
                suppress_tokens=None,
                **_pad_generation_kwargs(bundle.tokenizer),
                **generation_compile_kwargs(settings, bundle.text_model),
            )
        _synchronize_if_cuda(device)
        generation_elapsed = max(time.time() - generation_started, 1e-9)
        generated_rows = generate_ids[:, input_ids.shape[1] :]
        generated_token_count = max(0, int(generated_rows.numel()))
        stats = self._record_generation_stats(device, generated_token_count, generation_elapsed)
        captions = _decode_generated_rows(bundle.tokenizer, generated_rows, skip_special_tokens=False, strip_last=True)
        log_event(f"Alpha 1: batch generation complete. Token speed: {_generation_stats_text(stats)}.", self.config.title)
        return prompt_text, [self._postprocess(caption, settings) for caption in captions]

    @torch.inference_mode()
    def _generate_alpha_two_batch(self, images: Sequence[Image.Image], settings: dict[str, Any], bundle: LegacyModelBundle) -> tuple[str, list[str]]:
        prompt_text = self.build_alpha_two_prompt(
            str(settings.get("caption_type", "Descriptive")),
            settings.get("caption_length", "long"),
            settings.get("extra_options") or [],
            str(settings.get("name_input", "")),
            str(settings.get("custom_prompt", "")),
        )
        device = bundle.device
        pixel_values = torch.stack(
            [
                TVF.normalize(TVF.pil_to_tensor(image.resize((384, 384), Image.Resampling.LANCZOS)) / 255.0, [0.5], [0.5])
                for image in images
            ]
        ).to(device)
        batch_count = int(pixel_values.shape[0])
        with torch.amp.autocast(device_type="cuda", enabled=device.startswith("cuda")):
            vision_outputs = bundle.clip_model(pixel_values=pixel_values, output_hidden_states=True)
            embedded_images = bundle.image_adapter(vision_outputs.hidden_states)
        convo = [
            {"role": "system", "content": "You are a helpful image captioner."},
            {"role": "user", "content": prompt_text},
        ]
        if hasattr(bundle.tokenizer, "apply_chat_template"):
            convo_string = bundle.tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
        else:
            convo_string = prompt_text
        convo_tokens = bundle.tokenizer.encode(convo_string, return_tensors="pt", add_special_tokens=False, truncation=False).squeeze(0).to(device)
        prompt_tokens = bundle.tokenizer.encode(prompt_text, return_tensors="pt", add_special_tokens=False, truncation=False).squeeze(0).to(device)
        eot_id = bundle.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        eot_indices = (convo_tokens == eot_id).nonzero(as_tuple=True)[0].tolist() if isinstance(eot_id, int) else []
        preamble_len = eot_indices[1] - prompt_tokens.shape[0] if len(eot_indices) >= 2 else max(convo_tokens.shape[0] - prompt_tokens.shape[0], 0)
        embeds = _embed_tokens(bundle.text_model)
        convo_embeds = embeds(convo_tokens.unsqueeze(0))
        input_embeds = torch.cat(
            [
                convo_embeds[:, :preamble_len].expand(batch_count, -1, -1),
                embedded_images.to(dtype=convo_embeds.dtype),
                convo_embeds[:, preamble_len:].expand(batch_count, -1, -1),
            ],
            dim=1,
        ).to(device)
        input_ids = torch.cat(
            [
                convo_tokens[:preamble_len].unsqueeze(0).expand(batch_count, -1),
                torch.zeros((batch_count, embedded_images.shape[1]), dtype=torch.long, device=device),
                convo_tokens[preamble_len:].unsqueeze(0).expand(batch_count, -1),
            ],
            dim=1,
        )
        log_event(
            f"Alpha 2: generating {batch_count} caption(s) in one batch (max_new_tokens={int(settings.get('max_new_tokens', 300))}).",
            self.config.title,
        )
        self._record_generation_stats(device, 0, 0.0)
        _synchronize_if_cuda(device)
        generation_started = time.time()
        with attention_runtime_context(settings):
            generate_ids = bundle.text_model.generate(
                input_ids,
                inputs_embeds=input_embeds,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=int(settings.get("max_new_tokens", 300)),
                do_sample=True,
                suppress_tokens=None,
                **_pad_generation_kwargs(bundle.tokenizer),
                **generation_compile_kwargs(settings, bundle.text_model),
            )
        _synchronize_if_cuda(device)
        generation_elapsed = max(time.time() - generation_started, 1e-9)
        generated_rows = generate_ids[:, input_ids.shape[1] :]
        generated_token_count = max(0, int(generated_rows.numel()))
        stats = self._record_generation_stats(device, generated_token_count, generation_elapsed)
        captions = _decode_generated_rows(
            bundle.tokenizer,
            generated_rows,
            skip_special_tokens=False,
            strip_token_ids=[bundle.tokenizer.eos_token_id, eot_id if isinstance(eot_id, int) else None],
        )
        log_event(f"Alpha 2: batch generation complete. Token speed: {_generation_stats_text(stats)}.", self.config.title)
        return prompt_text, [self._postprocess(caption, settings) for caption in captions]

    def _postprocess(self, caption: str, settings: dict[str, Any]) -> str:
        caption = clean_legacy_caption(caption.strip(), aggressive=self.config.clean_aggressive)
        if settings.get("cut_off_sentence", True):
            caption = cut_off_last_sentence(caption)
        if settings.get("discard_repeating_sentences", True):
            caption = remove_repeating_sentences(caption)
        if settings.get("remove_newlines", True):
            caption = " ".join(caption.split())
        return caption.strip()

    def generate_batch(self, images: Sequence[Image.Image], settings: dict[str, Any], device_id: int | str | None = None) -> tuple[str, list[str]]:
        if not images:
            return "", []
        log_event(f"Caption batch request received ({len(images)} image(s)).", self.config.title)
        apply_torch_optimizations(settings, "before")
        selected_device = device_id if device_id is not None else first_device(settings.get("device_id", "0"), allow_cpu=True)
        bundle = self._bundle(
            selected_device,
            use_fp16=bool(settings.get("use_fp16", False)),
            use_4bit=bool(settings.get("use_4bit", settings.get("use_4bit_quantization", False))),
            low_cpu_mem_usage=bool(settings.get("low_cpu_mem_usage", False)),
            attention_backend=str(settings.get("attention_backend") or ("sdpa" if settings.get("use_sdpa_attention", False) else "auto")),
        )
        if self.config.mode == "pre_alpha":
            result = self._generate_pre_alpha_batch(images, settings, bundle)
            log_event("Caption batch request finished.", self.config.title)
            return result
        if self.config.mode == "alpha_one":
            result = self._generate_alpha_one_batch(images, settings, bundle)
            log_event("Caption batch request finished.", self.config.title)
            return result
        if self.config.mode == "alpha_two":
            result = self._generate_alpha_two_batch(images, settings, bundle)
            log_event("Caption batch request finished.", self.config.title)
            return result
        raise ValueError(f"Unknown legacy engine mode: {self.config.mode}")

    def generate(self, image: Image.Image, settings: dict[str, Any], device_id: int | str | None = None) -> tuple[str, str]:
        prompt, captions = self.generate_batch([image], settings, device_id=device_id)
        return prompt, captions[0] if captions else ""

    def caption_single(self, image_input: Any, settings: dict[str, Any]) -> CaptionResult:
        log_event("Single image caption started.", self.config.title)
        settings = dict(settings)
        settings["overwrite"] = False
        settings["append"] = False
        image_path = coerce_image_path(image_input, OUTPUTS_DIR / "temp")
        if image_path is None:
            raise ValueError("No input image selected.")
        if settings.get("use_subprocess", False):
            data = run_worker(
                "legacy_single",
                {
                    "variant": self.config.key,
                    "image_path": str(image_path),
                    "settings": settings,
                },
            )
            return CaptionResult(
                prompt=str(data.get("prompt") or ""),
                caption=str(data.get("caption") or ""),
                caption_path=Path(data["caption_path"]) if data.get("caption_path") else None,
                image_path=Path(data["image_path"]) if data.get("image_path") else None,
                elapsed=float(data.get("elapsed") or 0.0),
                details=str(data.get("details") or ""),
                metadata_path=Path(data["metadata_path"]) if data.get("metadata_path") else None,
            )
        start = time.time()
        reset_vram_peak_stats(parse_device_ids(settings.get("device_id", "0"), allow_cpu=True))
        before_vram = vram_usage_text()
        log_event(f"Loading image: {image_path}", self.config.title)
        image = load_rgb_image(image_path, int(settings.get("max_resolution", 1536) or 1536))
        prompt, caption = self.generate(image, settings)
        generation_stats = self.last_generation_stats
        apply_torch_optimizations(settings, "after")
        after_vram = vram_usage_text()
        log_event("Saving single image output.", self.config.title)
        final_caption = finalize_caption_text(
            caption,
            remove_newlines=bool(settings.get("remove_newlines", True)),
            prefix=str(settings.get("prefix", "")),
            suffix=str(settings.get("suffix", "")),
            replace_pairs=settings.get("replace_pairs"),
            replace_case_sensitive=bool(settings.get("replace_case_sensitive", False)),
            replace_single_word=bool(settings.get("replace_single_word", False)),
        )
        details = f"Token speed: {_generation_stats_text(generation_stats)}\n{optimization_status_text(settings)}\nBefore {before_vram}\nAfter {after_vram}"
        metadata = {
            "generation_type": "single_image",
            "engine": "legacy_siglip",
            "variant": self.config.key,
            "title": self.config.title,
            "checkpoint_dir": str(self.config.checkpoint_dir),
            "base_model_name": self.config.base_model_name,
            "clip_path": self.config.clip_path,
            "source_image_path": str(image_path),
            "prompt": prompt,
            "caption_raw": caption,
            "caption_final": final_caption,
            "settings": dict(settings),
            "elapsed_seconds": time.time() - start,
            "generated_tokens": generation_stats.generated_tokens,
            "generation_elapsed_seconds": generation_stats.elapsed_seconds,
            "tokens_per_second": generation_stats.tokens_per_second,
            "vram_before": before_vram,
            "vram_after": after_vram,
            "optimizations": optimization_status_text(settings),
        }
        actual_image, actual_caption, metadata_path, _run_dir = save_numbered_generation(
            image_path,
            final_caption,
            metadata,
            OUTPUTS_DIR,
            copy_image=self._should_save_image(settings),
        )
        log_event(f"Single image caption saved: {actual_caption}", self.config.title)
        return CaptionResult(
            prompt=prompt,
            caption=final_caption,
            caption_path=actual_caption,
            image_path=actual_image,
            elapsed=time.time() - start,
            details=details,
            metadata_path=metadata_path,
        )

    def _caption_path_for_batch(self, image_path: Path, input_root: Path, output_root: Path, preserve_subfolders: bool) -> tuple[Path, Path]:
        return resolve_output_paths(image_path, input_root, output_root, preserve_subfolders=preserve_subfolders)

    def batch_folder(self, settings: dict[str, Any]) -> Generator[str, None, None]:
        log_event("Folder batch requested.", self.config.title)
        if settings.get("use_subprocess", False):
            yield f"Starting {self.config.title} subprocess batch. The child process will exit when the run ends."
            data = run_worker(
                "legacy_batch",
                {
                    "variant": self.config.key,
                    "settings": settings,
                },
            )
            yield str(data.get("progress") or "Subprocess batch completed.")
            return

        input_folder = Path(str(settings.get("input_folder", "")).strip())
        output_folder_text = str(settings.get("output_folder", "")).strip()
        output_folder = Path(output_folder_text) if output_folder_text else input_folder
        include_subfolders = bool(settings.get("process_subfolders", True))
        overwrite = bool(settings.get("overwrite", False))
        append = bool(settings.get("append", False))
        preserve_subfolders = include_subfolders

        if not input_folder.is_dir():
            yield f"Input folder not found: {input_folder}"
            return
        output_folder.mkdir(parents=True, exist_ok=True)
        log_event(f"Scanning folder: {input_folder}", self.config.title)

        all_images = discover_images(input_folder, include_subfolders=include_subfolders)
        images: list[Path] = []
        skipped_initial = 0
        for image_path in all_images:
            _, caption_path = self._caption_path_for_batch(image_path, input_folder, output_folder, preserve_subfolders)
            if caption_path.exists() and not overwrite and not append:
                skipped_initial += 1
                continue
            images.append(image_path)

        if not images:
            yield f"No images to process. Found {len(all_images)} image files, all skipped or unsupported."
            return

        self.stop_flag.reset()
        devices = parse_device_ids(settings.get("gpu_ids") or settings.get("device_id") or "0", allow_cpu=True)
        reset_vram_peak_stats(devices)
        batch_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        batch_size = max(1, int(settings.get("batch_size", 1) or 1))
        progress_text = (
            f"Starting {self.config.title} folder batch.\n"
            f"Input: {input_folder}\n"
            f"Output: {output_folder}\n"
            f"Images queued: {len(images)} of {len(all_images)} found\n"
            f"Skipped existing: {skipped_initial}\n"
            f"Devices: {', '.join(str(d) for d in devices)}\n"
            f"Batch size per device: {batch_size}\n"
        )
        log_event(f"Folder batch started: {len(images)} queued on {len(devices)} device(s).", self.config.title)
        yield progress_text

        chunks = split_round_robin(images, devices)
        aggregate_lock = threading.Lock()
        aggregate = {"processed": 0, "skipped": skipped_initial, "failed": 0}
        total_generated_tokens = 0
        total_generation_seconds = 0.0
        started_all = time.time()
        total_found = len(all_images)
        process_chunks = [
            (device_id, chunk)
            for device_id, chunk in zip(devices, chunks)
            if chunk
        ]

        if len(process_chunks) > 1:
            ctx = mp.get_context("spawn")
            process_queue: Any = ctx.Queue()
            processes: list[mp.Process] = []
            chunk_sizes: dict[str, int] = {}
            done_devices: set[str] = set()
            for device_id, chunk in process_chunks:
                chunk_sizes[str(device_id)] = len(chunk)
                process = ctx.Process(
                    target=_legacy_folder_process_worker,
                    args=(
                        process_queue,
                        self.config.key,
                        [str(path) for path in chunk],
                        str(input_folder),
                        str(output_folder),
                        preserve_subfolders,
                        dict(settings),
                        device_id,
                    ),
                    daemon=False,
                )
                processes.append(process)
                log_event(f"Starting process worker for device {device_id}: {len(chunk)} image(s).", self.config.title)
                process.start()

            try:
                done_count = 0
                while done_count < len(processes):
                    if self.stop_flag.value:
                        for process in processes:
                            if process.is_alive():
                                process.terminate()
                        yield throttle_status("Batch processing stopped. Child GPU processes were terminated.", progress_text)
                        break
                    try:
                        event = process_queue.get(timeout=0.5)
                    except queue.Empty:
                        if not any(process.is_alive() for process in processes):
                            break
                        continue

                    kind = str(event.get("kind") or "")
                    device_id = event.get("device_id")
                    if kind == "progress":
                        with aggregate_lock:
                            aggregate["processed"] += int(event.get("processed_delta", 0) or 0)
                            aggregate["skipped"] += int(event.get("skipped_delta", 0) or 0)
                            aggregate["failed"] += int(event.get("failed_delta", 0) or 0)
                            total_generated_tokens += int(event.get("generated_tokens", 0) or 0)
                            total_generation_seconds += float(event.get("generation_seconds", 0.0) or 0.0)
                            token_speed = total_generated_tokens / max(total_generation_seconds, 1e-9)
                            line = batch_progress_line(
                                processed=aggregate["processed"],
                                total=total_found,
                                skipped=aggregate["skipped"],
                                failed=aggregate["failed"],
                                started=started_all,
                                last_batch_count=int(event.get("last_batch_count", 0) or 0),
                                last_batch_seconds=float(event.get("last_batch_seconds", 0.0) or 0.0),
                                token_speed=token_speed,
                                device_id=device_id,
                                worker_processed=int(event.get("local_processed", 0) or 0),
                                worker_total=int(event.get("worker_total", 0) or 0),
                                worker_skipped=int(event.get("local_skipped", 0) or 0),
                                worker_failed=int(event.get("local_failed", 0) or 0),
                            )
                        if event.get("message"):
                            line = f"{line} {event['message']}"
                        log_event(line, self.config.title)
                        progress_text = throttle_status(line, progress_text)
                        yield progress_text
                    elif kind == "done":
                        key = str(device_id)
                        if key not in done_devices:
                            done_devices.add(key)
                            done_count += 1
                        line = (
                            f"Device {device_id}: process complete with "
                            f"{int(event.get('local_processed', 0) or 0)} processed, "
                            f"{int(event.get('local_skipped', 0) or 0)} skipped, "
                            f"{int(event.get('local_failed', 0) or 0)} failed."
                        )
                        log_event(line, self.config.title)
                        progress_text = throttle_status(line, progress_text)
                        yield progress_text
                    elif kind == "fatal":
                        key = str(device_id)
                        if key not in done_devices:
                            done_devices.add(key)
                            done_count += 1
                            with aggregate_lock:
                                aggregate["failed"] += chunk_sizes.get(key, 0)
                        line = f"Device {device_id}: process failed: {event.get('error') or 'unknown error'}"
                        log_event(line, self.config.title)
                        progress_text = throttle_status(line, progress_text)
                        yield progress_text

                for process in processes:
                    process.join(timeout=5.0)
                for device_id, process in zip([device for device, _chunk in process_chunks], processes):
                    key = str(device_id)
                    if process.exitcode not in (0, None) and key not in done_devices:
                        with aggregate_lock:
                            aggregate["failed"] += chunk_sizes.get(key, 0)
                        line = f"Device {device_id}: process exited with code {process.exitcode}."
                        log_event(line, self.config.title)
                        progress_text = throttle_status(line, progress_text)
                        yield progress_text
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=2.0)

            final = "Batch processing stopped." if self.stop_flag.value else "Batch processing complete."
            with aggregate_lock:
                token_speed = total_generated_tokens / max(total_generation_seconds, 1e-9)
                token_detail = f"{total_generated_tokens} token(s) in {total_generation_seconds:.2f}s ({token_speed:.2f} tok/s)"
                final_line = (
                    f"{final} Processed {aggregate['processed']}/{total_found}, "
                    f"{aggregate['skipped']} skipped, {aggregate['failed']} failed.\n"
                    f"Token speed: {token_detail}"
                )
            yield throttle_status(final_line, progress_text)
            return

        def worker(device_id: int | str, chunk: list[Path]) -> None:
            nonlocal total_generated_tokens, total_generation_seconds
            log_event(f"Batch worker started on device {device_id}: {len(chunk)} image(s).", self.config.title)
            local_processed = 0
            local_failed = 0
            local_skipped = 0
            for offset in range(0, len(chunk), batch_size):
                if self.stop_flag.value:
                    batch_queue.put(("progress", f"Device {device_id}: stop requested."))
                    break
                batch_paths = chunk[offset : offset + batch_size]
                work_items: list[tuple[Path, Path, Path]] = []
                with aggregate_lock:
                    for image_path in batch_paths:
                        output_image_path, caption_path = self._caption_path_for_batch(image_path, input_folder, output_folder, preserve_subfolders)
                        if caption_path.exists() and not overwrite and not append:
                            aggregate["skipped"] += 1
                            local_skipped += 1
                            continue
                        work_items.append((image_path, output_image_path, caption_path))
                if not work_items:
                    with aggregate_lock:
                        line = batch_progress_line(
                            processed=aggregate["processed"],
                            total=total_found,
                            skipped=aggregate["skipped"],
                            failed=aggregate["failed"],
                            started=started_all,
                            last_batch_count=0,
                            last_batch_seconds=0.0,
                            token_speed=total_generated_tokens / max(total_generation_seconds, 1e-9),
                            device_id=device_id if len(devices) > 1 else None,
                            worker_processed=local_processed,
                            worker_total=len(chunk),
                            worker_skipped=local_skipped,
                            worker_failed=local_failed,
                        )
                    batch_queue.put(("progress", line))
                    continue
                try:
                    batch_started = time.time()
                    names = ", ".join(path.name for path, _, _ in work_items[:3])
                    if len(work_items) > 3:
                        names += ", ..."
                    log_event(
                        f"Device {device_id}: captioning true batch {offset + 1}-{offset + len(work_items)} of {len(chunk)} ({names}).",
                        self.config.title,
                    )
                    loaded_images = [
                        load_rgb_image(image_path, int(settings.get("max_resolution", 1536) or 1536))
                        for image_path, _, _ in work_items
                    ]
                    _prompt, captions = self.generate_batch(loaded_images, settings, device_id=device_id)
                    stats = self._generation_stats_for_device(device_id)
                    batch_elapsed = max(time.time() - batch_started, 1e-9)
                    with aggregate_lock:
                        total_generated_tokens += stats.generated_tokens
                        total_generation_seconds += stats.elapsed_seconds
                        token_speed = total_generated_tokens / max(total_generation_seconds, 1e-9)
                        saved_count = 0
                        for (image_path, output_image_path, caption_path), caption in zip(work_items, captions):
                            actual_caption = save_caption_file(
                                caption_path,
                                caption,
                                overwrite=overwrite,
                                append=append,
                                remove_newlines=bool(settings.get("remove_newlines", True)),
                                prefix=str(settings.get("prefix", "")),
                                suffix=str(settings.get("suffix", "")),
                                replace_pairs=settings.get("replace_pairs"),
                                replace_case_sensitive=bool(settings.get("replace_case_sensitive", False)),
                                replace_single_word=bool(settings.get("replace_single_word", False)),
                            )
                            copy_image_if_needed(image_path, output_image_path, self._should_save_image(settings))
                            if actual_caption:
                                aggregate["processed"] += 1
                                local_processed += 1
                                saved_count += 1
                                log_event(f"Device {device_id}: saved {actual_caption}.", self.config.title)
                            else:
                                aggregate["skipped"] += 1
                                local_skipped += 1
                        line = batch_progress_line(
                            processed=aggregate["processed"],
                            total=total_found,
                            skipped=aggregate["skipped"],
                            failed=aggregate["failed"],
                            started=started_all,
                            last_batch_count=saved_count,
                            last_batch_seconds=batch_elapsed,
                            token_speed=token_speed,
                            device_id=device_id if len(devices) > 1 else None,
                            worker_processed=local_processed,
                            worker_total=len(chunk),
                            worker_skipped=local_skipped,
                            worker_failed=local_failed,
                        )
                except Exception as exc:
                    with aggregate_lock:
                        aggregate["failed"] += len(work_items)
                        local_failed += len(work_items)
                        line = batch_progress_line(
                            processed=aggregate["processed"],
                            total=total_found,
                            skipped=aggregate["skipped"],
                            failed=aggregate["failed"],
                            started=started_all,
                            last_batch_count=0,
                            last_batch_seconds=0.0,
                            token_speed=total_generated_tokens / max(total_generation_seconds, 1e-9),
                            device_id=device_id if len(devices) > 1 else None,
                            worker_processed=local_processed,
                            worker_total=len(chunk),
                            worker_skipped=local_skipped,
                            worker_failed=local_failed,
                        )
                    line = f"{line} Failed batch: {format_exception(exc)}"
                batch_queue.put(("progress", line))
            batch_queue.put(("done", f"Device {device_id}: complete with {local_processed} processed, {local_skipped} skipped, {local_failed} failed."))

        threads = [
            threading.Thread(target=worker, args=(device_id, chunk), daemon=True)
            for device_id, chunk in zip(devices, chunks)
            if chunk
        ]
        for thread in threads:
            thread.start()

        done_count = 0
        while done_count < len(threads):
            try:
                kind, line = batch_queue.get(timeout=0.5)
            except queue.Empty:
                if not any(thread.is_alive() for thread in threads):
                    break
                continue
            log_event(line, self.config.title)
            progress_text = throttle_status(line, progress_text)
            if kind == "done":
                done_count += 1
            yield progress_text

        for thread in threads:
            thread.join(timeout=1.0)
        final = "Batch processing stopped." if self.stop_flag.value else "Batch processing complete."
        with aggregate_lock:
            token_speed = total_generated_tokens / max(total_generation_seconds, 1e-9)
            token_detail = f"{total_generated_tokens} token(s) in {total_generation_seconds:.2f}s ({token_speed:.2f} tok/s)"
            final_line = (
                f"{final} Processed {aggregate['processed']}/{total_found}, "
                f"{aggregate['skipped']} skipped, {aggregate['failed']} failed.\n"
                f"Token speed: {token_detail}"
            )
        yield throttle_status(final_line, progress_text)


def _legacy_engine_for_key(variant: str, base_dir: Path) -> LegacySiglipEngine:
    if variant == "pre_alpha":
        return create_pre_alpha_engine(base_dir)
    if variant == "alpha_one":
        return create_alpha_one_engine(base_dir)
    if variant == "alpha_two":
        return create_alpha_two_engine(base_dir)
    raise ValueError(f"Unknown legacy variant: {variant}")


def _legacy_folder_process_worker(
    event_queue: Any,
    variant: str,
    path_texts: list[str],
    input_folder_text: str,
    output_folder_text: str,
    preserve_subfolders: bool,
    settings: dict[str, Any],
    device_id: int | str,
) -> None:
    engine = _legacy_engine_for_key(variant, OUTPUTS_DIR.parent)
    paths = [Path(path) for path in path_texts]
    input_folder = Path(input_folder_text)
    output_folder = Path(output_folder_text)
    settings = dict(settings)
    settings["gpu_ids"] = str(device_id)
    settings["device_id"] = str(device_id)
    settings["use_subprocess"] = False
    overwrite = bool(settings.get("overwrite", False))
    append = bool(settings.get("append", False))
    batch_size = max(1, int(settings.get("batch_size", 1) or 1))
    max_resolution = int(settings.get("max_resolution", 1536) or 1536)
    local_processed = 0
    local_skipped = 0
    local_failed = 0
    try:
        log_event(f"Device {device_id}: process worker started with {len(paths)} image(s).", engine.config.title)
        for offset in range(0, len(paths), batch_size):
            batch_paths = paths[offset : offset + batch_size]
            work_items: list[tuple[Path, Path, Path]] = []
            for image_path in batch_paths:
                output_image_path, caption_path = engine._caption_path_for_batch(
                    image_path,
                    input_folder,
                    output_folder,
                    preserve_subfolders,
                )
                if caption_path.exists() and not overwrite and not append:
                    local_skipped += 1
                    continue
                work_items.append((image_path, output_image_path, caption_path))

            if not work_items:
                event_queue.put(
                    {
                        "kind": "progress",
                        "device_id": device_id,
                        "processed_delta": 0,
                        "skipped_delta": len(batch_paths),
                        "failed_delta": 0,
                        "local_processed": local_processed,
                        "local_skipped": local_skipped,
                        "local_failed": local_failed,
                        "worker_total": len(paths),
                        "last_batch_count": 0,
                        "last_batch_seconds": 0.0,
                        "generated_tokens": 0,
                        "generation_seconds": 0.0,
                    }
                )
                continue

            try:
                names = ", ".join(path.name for path, _, _ in work_items[:3])
                if len(work_items) > 3:
                    names += ", ..."
                log_event(
                    f"Device {device_id}: process captioning true batch {offset + 1}-{offset + len(work_items)} of {len(paths)} ({names}).",
                    engine.config.title,
                )
                batch_started = time.time()
                loaded_images = [load_rgb_image(image_path, max_resolution) for image_path, _, _ in work_items]
                _prompt, captions = engine.generate_batch(loaded_images, settings, device_id=device_id)
                stats = engine._generation_stats_for_device(device_id)
                batch_elapsed = max(time.time() - batch_started, 1e-9)
                saved_count = 0
                skipped_count = 0
                for (image_path, output_image_path, caption_path), caption in zip(work_items, captions):
                    actual_caption = save_caption_file(
                        caption_path,
                        caption,
                        overwrite=overwrite,
                        append=append,
                        remove_newlines=bool(settings.get("remove_newlines", True)),
                        prefix=str(settings.get("prefix", "")),
                        suffix=str(settings.get("suffix", "")),
                        replace_pairs=settings.get("replace_pairs"),
                        replace_case_sensitive=bool(settings.get("replace_case_sensitive", False)),
                        replace_single_word=bool(settings.get("replace_single_word", False)),
                    )
                    copy_image_if_needed(image_path, output_image_path, engine._should_save_image(settings))
                    if actual_caption:
                        saved_count += 1
                        local_processed += 1
                        log_event(f"Device {device_id}: process saved {actual_caption}.", engine.config.title)
                    else:
                        skipped_count += 1
                        local_skipped += 1
                event_queue.put(
                    {
                        "kind": "progress",
                        "device_id": device_id,
                        "processed_delta": saved_count,
                        "skipped_delta": skipped_count,
                        "failed_delta": 0,
                        "local_processed": local_processed,
                        "local_skipped": local_skipped,
                        "local_failed": local_failed,
                        "worker_total": len(paths),
                        "last_batch_count": saved_count,
                        "last_batch_seconds": batch_elapsed,
                        "generated_tokens": stats.generated_tokens,
                        "generation_seconds": stats.elapsed_seconds,
                    }
                )
            except Exception as exc:
                failed_count = len(work_items)
                local_failed += failed_count
                event_queue.put(
                    {
                        "kind": "progress",
                        "device_id": device_id,
                        "processed_delta": 0,
                        "skipped_delta": 0,
                        "failed_delta": failed_count,
                        "local_processed": local_processed,
                        "local_skipped": local_skipped,
                        "local_failed": local_failed,
                        "worker_total": len(paths),
                        "last_batch_count": 0,
                        "last_batch_seconds": 0.0,
                        "generated_tokens": 0,
                        "generation_seconds": 0.0,
                        "message": f"Failed batch: {format_exception(exc)}",
                    }
                )
        event_queue.put(
            {
                "kind": "done",
                "device_id": device_id,
                "local_processed": local_processed,
                "local_skipped": local_skipped,
                "local_failed": local_failed,
            }
        )
    except Exception as exc:
        event_queue.put({"kind": "fatal", "device_id": device_id, "error": format_exception(exc)})
    finally:
        try:
            engine._bundles.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def create_pre_alpha_engine(base_dir: Path) -> LegacySiglipEngine:
    return LegacySiglipEngine(
        LegacyVariantConfig(
            key="pre_alpha",
            title="Joy Caption Pre Alpha",
            checkpoint_dir=base_dir / "model_files_pre_alpha",
            mode="pre_alpha",
            default_dtype=torch.bfloat16,
            clean_aggressive=False,
        )
    )


def create_alpha_one_engine(base_dir: Path) -> LegacySiglipEngine:
    return LegacySiglipEngine(
        LegacyVariantConfig(
            key="alpha_one",
            title="Joy Caption Alpha 1",
            checkpoint_dir=base_dir / "model_files_alpha_one",
            mode="alpha_one",
            default_dtype=torch.bfloat16,
            clean_aggressive=True,
        )
    )


def create_alpha_two_engine(base_dir: Path) -> LegacySiglipEngine:
    return LegacySiglipEngine(
        LegacyVariantConfig(
            key="alpha_two",
            title="Joy Caption Alpha 2",
            checkpoint_dir=base_dir / "model_files_alpha_two",
            mode="alpha_two",
            default_dtype=torch.bfloat16,
            clean_aggressive=False,
        )
    )
