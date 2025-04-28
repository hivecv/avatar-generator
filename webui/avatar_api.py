import os
import time
import traceback
from enum import Enum
from io import BytesIO
from pathlib import Path

from PIL import Image
import webuiapi
from fastapi import FastAPI, File, Request, Response, UploadFile, HTTPException

if 'POSTHOG_KEY' in os.environ and os.environ['POSTHOG_KEY'] not in [None, '']:
    from posthog import Posthog
    posthog = Posthog(project_api_key=os.environ['POSTHOG_KEY'], host='https://eu.i.posthog.com')
else:
    class FakeHog:
        def __getattr__(self, *args, **kwargs):
            return lambda *args, **kwargs: print(args, kwargs)
    posthog = FakeHog()

app = FastAPI()
client = webuiapi.WebUIApi(host="100.104.185.52")
assets_path = Path(__file__).parent / "assets"

WEBUI_OPTIONS = {
    # "forge_additional_modules": [
    #     '/stable-diffusion-webui/models/text_encoder/clip_l.safetensors',
    #     '/stable-diffusion-webui/models/text_encoder/clip_g.safetensors',
    #     '/stable-diffusion-webui/models/text_encoder/tx5xxl_fp16.safetensors'
    # ],
    "forge_additional_modules": [],
    # "sd_model_checkpoint": "sd3.5_large_turbo.safetensors",
    # "sd_model_checkpoint": "v1-5-pruned-emaonly.ckpt",
    "sd_model_checkpoint": "sd-v1-5-inpainting.ckpt",
    "sd_checkpoints_keep_in_cpu": False,
    "sd_checkpoint_cache": 1,
    "grid_save": False,
    "save_write_log_csv": False,
    "samples_save": False,
    "save_to_dirs": False,
    "grid_save_to_dirs": False,
    "pin_memory": True,
}

def generate_avatar_image(source_file, mask_file, face_file):
    return client.img2img(
        images=[Image.open(face_file)],
        mask_image=Image.open(mask_file),
        inpainting_fill=1,
        prompt='Extreme details, high resolution, best quality, portrait warm light',
        # sampler_name='SGM Uniform',
        # scheduler='Euler',
        steps=10,
        seed=1,
        image_cfg_scale=1.5,
        cfg_scale=3.5,
        denoising_strength=0.3,
        restore_faces=True,
        do_not_save_grid=True,
        do_not_save_samples=True,
        resize_mode=2,
        width=1024,
        height=1024,
        reactor=webuiapi.ReActor(
            img=Image.open(source_file),
            enable=True,
            # face_restorer_name='CodeFormer',
            # face_restorer_visibility=1,
            # swap_in_source=False,
            # swap_in_generated=True,
            # codeFormer_weight=0.8,
            # gender_source=2 ,
            # gender_target=2,
        )
    ).image

for i in range(10):
    try:
        client.set_options(WEBUI_OPTIONS)
        generate_avatar_image(
            source_file=assets_path.joinpath(f"basic_jan.png"),
            face_file=assets_path.joinpath(f"basic_jan.png"),
            mask_file=assets_path.joinpath(f"basic_jan_mask.png")
        )
        client.set_options(WEBUI_OPTIONS)
        break
    except Exception as e:
        print("Failure in generation, retrying...")
        time.sleep(12)
        if i == 9:
            raise

print(client.get_options())

class GenerateAvatarTypes(str, Enum):
    basic_jan = "basic_jan"
    botanist_jan = "botanist_jan"
    scientist_jan = "scientist_jan"

@app.post("/generate/{avatar_type}", responses={200: {"content": {"image/png": {}}}}, response_class=Response)
async def generate(avatar_type: GenerateAvatarTypes, request: Request, file: UploadFile = File(...)):
    posthog.capture(request.client.host, '$pageview', {'$current_url': str(request.url)})
    start = time.time()
    try:
        face_file = assets_path.joinpath(f"{avatar_type.value}.png")
        mask_file = assets_path.joinpath(f"{avatar_type.value}_mask.png")

        if not face_file.exists() or not mask_file.exists():
            raise HTTPException(status_code=404, detail=f"Avatar {avatar_type.value} not found")

        result = generate_avatar_image(source_file=file.file, mask_file=mask_file, face_file=face_file)
        result_bytes = BytesIO()
        result.save(result_bytes, "PNG")
        result_bytes.seek(0)
        content = result_bytes.read()
        try:
            return Response(
                media_type="image/png",
                content=content,
            )
        finally:
            posthog.capture(request.client.host, "request_complete",
                {
                    "avatar_type": avatar_type.value,
                    "result_size": len(content),
                    "processing_time": time.time() - start,
                }
            )
    except Exception as e:
        posthog.capture(request.client.host, "request_error",
            {
                "error_str": str(e),
                "traceback_str": traceback.format_exc(),
                "processing_time": time.time() - start,
            }
        )
        raise

class FinalizeAvatarTypes(str, Enum):
    basic_jan = "basic_jan"
    botanist_jan = "botanist_jan"
    scientist_jan = "scientist_jan"
    me = "me"

@app.post("/finalize/{avatar_type}", responses={200: {"content": {"image/png": {}}}}, response_class=Response)
async def finalize(avatar_type: FinalizeAvatarTypes, request: Request, file: UploadFile = File(...)):
    posthog.capture(request.client.host, '$pageview', {'$current_url': str(request.url)})
    start = time.time()
    try:
        background = Image.open(assets_path.joinpath("space_suit_overlay_back.png"))
        foreground = Image.open(assets_path.joinpath("space_suit_overlay_front.png"))
        if avatar_type == "me":
            result = Image.open(file.file)
        else:
            mask_file = assets_path.joinpath(f"{avatar_type}_mask_suited.png")
            if not mask_file.exists():
                raise HTTPException(status_code=404, detail=f"Avatar {avatar_type} not found")
            mask = Image.open(mask_file).convert('L')
            request_file = Image.open(file.file)
            request_file.putalpha(mask)
            result = Image.alpha_composite(background, Image.alpha_composite(request_file, foreground))

        result_bytes = BytesIO()
        result.save(result_bytes, "PNG")
        result_bytes.seek(0)
        content = result_bytes.read()
        try:
            return Response(
                media_type="image/png",
                content=content,
            )
        finally:
            posthog.capture(request.client.host, "request_complete",
                {
                    "avatar_type": avatar_type,
                    "result_size": len(content),
                    "processing_time": time.time() - start,
                }
            )
    except Exception as e:
        posthog.capture(request.client.host, "request_error",
            {
                "error_str": str(e),
                "traceback_str": traceback.format_exc(),
                "processing_time": time.time() - start,
            }
        )
        raise
