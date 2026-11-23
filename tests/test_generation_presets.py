
"""Tests for model-aware rendering defaults."""

from openimagegen.models.presets import generation_profiles, generation_templates, recommended_generation_params


def test_sd15_quality_profile_preserves_native_resolution():
    profiles = generation_profiles("dreamshaper_8", {
        "name": "Lykon/dreamshaper-8", "type": "stable_diffusion",
    })
    assert profiles["recommended"]["width"] == 512
    assert profiles["quality"]["num_inference_steps"] > profiles["draft"]["num_inference_steps"]
    assert profiles["quality"]["sampler"] == "dpmpp_2m_sde_karras"
    assert profiles["recommended"]["num_inference_steps"] == 25


def test_distilled_and_flow_models_receive_compatible_settings():
    turbo = recommended_generation_params("sd_turbo", {
        "name": "stabilityai/sd-turbo", "type": "stable_diffusion",
    })
    flux = recommended_generation_params("flux_schnell", {
        "name": "black-forest-labs/FLUX.1-schnell", "type": "flux",
    })
    assert turbo["guidance_scale"] == 0
    assert turbo["num_inference_steps"] == 2
    assert turbo["sampler"] == "euler_a"
    assert flux["guidance_scale"] == 0
    assert flux["num_inference_steps"] == 4
    assert flux["sampler"] == "flowmatch_euler"


def test_remote_profiles_do_not_request_a_local_scheduler():
    profile = recommended_generation_params("hosted", {
        "name": "gpt-image-1", "type": "openai_image",
    })
    assert "sampler" not in profile
    assert profile["num_inference_steps"] == 1


def test_model_cards_drive_specialized_presets():
    sdxl_turbo = recommended_generation_params("sdxl_turbo", {
        "name": "stabilityai/sdxl-turbo", "type": "stable_diffusion_xl",
    })
    qwen = recommended_generation_params("qwen_image", {
        "name": "Qwen/Qwen-Image", "type": "qwen_image",
    })
    sana = recommended_generation_params("sana", {
        "name": "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers", "type": "sana",
    })
    assert (sdxl_turbo["width"], sdxl_turbo["height"]) == (512, 512)
    assert sdxl_turbo["guidance_scale"] == 0
    assert qwen["num_inference_steps"] == 50
    assert sana["guidance_scale"] == 4.5

    turbo = recommended_generation_params("qwen_image_2512_turbo", {
        "name": "Qwen/Qwen-Image-2512-Turbo", "type": "qwen_image",
    })
    assert turbo["num_inference_steps"] == 8
    assert turbo["guidance_scale"] == 1.0


def test_templates_are_complete_and_have_declared_model_families():
    templates = generation_templates()
    assert len(templates) >= 8
    assert len({template["id"] for template in templates}) == len(templates)
    for template in templates:
        assert template["name"] and template["prompt"]
        assert template["families"]
        assert template["width"] % 8 == template["height"] % 8 == 0
