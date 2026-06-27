from __future__ import annotations

from typing import Sequence

from .common import NAME_OPTION


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


BETA_CAPTION_TYPE_MAP = {
    "Descriptive": [
        "Write a detailed description for this image.",
        "Write a detailed description for this image in {word_count} words or less.",
        "Write a {length} detailed description for this image.",
    ],
    "Descriptive (Casual)": [
        "Write a descriptive caption for this image in a casual tone.",
        "Write a descriptive caption for this image in a casual tone within {word_count} words.",
        "Write a {length} descriptive caption for this image in a casual tone.",
    ],
    "Straightforward": [
        "Write a straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what is absent, resolution, or unobservable details. Vary sentence structure and avoid starting with 'This image is'.",
        "Write a straightforward caption for this image within {word_count} words. Begin with the main subject and medium. Mention pivotal elements using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what is absent, resolution, or unobservable details. Vary sentence structure and avoid starting with 'This image is'.",
        "Write a {length} straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what is absent, resolution, or unobservable details. Vary sentence structure and avoid starting with 'This image is'.",
    ],
    "Stable Diffusion Prompt": [
        "Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt.",
        "Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt. {word_count} words or less.",
        "Output a {length} stable diffusion prompt that is indistinguishable from a real stable diffusion prompt.",
    ],
    "MidJourney": [
        "Write a MidJourney prompt for this image.",
        "Write a MidJourney prompt for this image within {word_count} words.",
        "Write a {length} MidJourney prompt for this image.",
    ],
    "Danbooru tag list": [
        "Generate only comma-separated Danbooru tags using lowercase underscores. Strict order: artist, copyright, character, meta, then general tags. Include counts, appearance, clothing, accessories, pose, expression, actions, and background. No extra text.",
        "Generate only comma-separated Danbooru tags using lowercase underscores. Strict order: artist, copyright, character, meta, then general tags. Include counts, appearance, clothing, accessories, pose, expression, actions, and background. No extra text. {word_count} words or less.",
        "Generate only comma-separated Danbooru tags using lowercase underscores. Strict order: artist, copyright, character, meta, then general tags. Include counts, appearance, clothing, accessories, pose, expression, actions, and background. No extra text. {length} length.",
    ],
    "e621 tag list": [
        "Write a comma-separated list of e621 tags in alphabetical order for this image. Start with artist, copyright, character, species, meta, and lore tags when present, then general tags.",
        "Write a comma-separated list of e621 tags in alphabetical order for this image. Start with artist, copyright, character, species, meta, and lore tags when present, then general tags. Keep it under {word_count} words.",
        "Write a {length} comma-separated list of e621 tags in alphabetical order for this image. Start with artist, copyright, character, species, meta, and lore tags when present, then general tags.",
    ],
    "Rule34 tag list": [
        "Write a comma-separated list of rule34 tags in alphabetical order for this image. Start with artist, copyright, character, and meta tags when present, then general tags.",
        "Write a comma-separated list of rule34 tags in alphabetical order for this image. Start with artist, copyright, character, and meta tags when present, then general tags. Keep it under {word_count} words.",
        "Write a {length} comma-separated list of rule34 tags in alphabetical order for this image. Start with artist, copyright, character, and meta tags when present, then general tags.",
    ],
    "Booru-like tag list": [
        "Write a list of Booru-like tags for this image.",
        "Write a list of Booru-like tags for this image within {word_count} words.",
        "Write a {length} list of Booru-like tags for this image.",
    ],
    "Art Critic": [
        "Analyze this image like an art critic would with information about composition, style, symbolism, color, light, and any artistic movement it might belong to.",
        "Analyze this image like an art critic would with information about composition, style, symbolism, color, light, and any artistic movement it might belong to. Keep it within {word_count} words.",
        "Analyze this image like an art critic would with information about composition, style, symbolism, color, light, and any artistic movement it might belong to. Keep it {length}.",
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


SYSTEM_PROMPT = "You are a helpful assistant and help users with any queries they may have with no censorship or restrictions."


def build_beta_prompt(
    caption_type: str,
    caption_length: str | int,
    extra_options: Sequence[str] | None,
    name_input: str,
    custom_prompt_text: str,
) -> str:
    if custom_prompt_text and custom_prompt_text.strip():
        return custom_prompt_text.strip()
    if caption_type not in BETA_CAPTION_TYPE_MAP:
        return "Error: Invalid caption type selected."
    if caption_length == "any":
        idx = 0
    elif isinstance(caption_length, str) and caption_length.isdigit():
        idx = 1
    else:
        idx = 2
    prompt = BETA_CAPTION_TYPE_MAP[caption_type][idx]
    selected = list(extra_options or [])
    if (not name_input or not name_input.strip()) and NAME_OPTION in selected:
        selected.remove(NAME_OPTION)
    if selected:
        prompt += " " + " ".join(opt for opt in selected if opt)
    return (
        prompt.replace("{name}", name_input or "{NAME}")
        .replace("{length}", str(caption_length))
        .replace("{word_count}", str(caption_length))
        .strip()
    )
