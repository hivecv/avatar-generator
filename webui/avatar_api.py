import os
import time
import traceback
from enum import Enum
from io import BytesIO
from pathlib import Path
import fastface as ff
import numpy as np
from PIL import Image, ImageOps
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
fd_model = ff.FaceDetector.from_pretrained("lffd_original")
fd_model.eval()
fd_model.to("cuda")
client = webuiapi.WebUIApi()
assets_path = Path(__file__).parent / "assets"

PERFORMANCE_WEBUI_OPTIONS = {
    "forge_additional_modules": [],
    "sd_model_checkpoint": "sd-v1-5-inpainting.ckpt",
}

QUALITY_WEBUI_OPTIONS = {
    "forge_additional_modules": [
        '/stable-diffusion-webui/models/text_encoder/clip_l.safetensors',
        '/stable-diffusion-webui/models/text_encoder/clip_g.safetensors',
        '/stable-diffusion-webui/models/text_encoder/tx5xxl_fp16.safetensors'
    ],
    "sd_model_checkpoint": "sd3.5_large_turbo.safetensors",
}

def generate_performance_avatar_image(source_file, mask_file, face_file):
    result = client.img2img(
        images=[Image.open(face_file)],
        mask_image=Image.open(mask_file),
        inpainting_fill=1,
        prompt='Extreme details, high resolution, best quality, portrait warm light',
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
        )
    ).image
    return result

def generate_quality_avatar_image(source_file, mask_file, face_file):
    client.set_options(QUALITY_WEBUI_OPTIONS)
    result = client.img2img(
        images=[Image.open(face_file)],
        mask_image=Image.open(mask_file),
        inpainting_fill=1,
        prompt='Ultra-realistic portrait, 8k resolution, high dynamic range, highly detailed facial features, natural skin texture, glossy eyes, fine wrinkles and pores, subsurface scattering, realistic lighting, soft shadows, sharp focus, professional studio photography, photorealistic',
        sampler_name='SGM Uniform',
        negative_prompt='blurry, low resolution, bad anatomy, distorted face, unnatural lighting, overexposed, unrealistic skin texture, oversaturation, artifacts, weird shadows, poorly blended, visible brushstrokes, cartoonish, plastic skin, unrealistic proportions, low quality',
        scheduler='euler',
        steps=30,
        seed=1,
        image_cfg_scale=1.5,
        cfg_scale=4.5,
        denoising_strength=0.78,
        restore_faces=True,
        do_not_save_grid=True,
        do_not_save_samples=True,
        resize_mode=4,
        width=1024,
        height=1024,
        reactor=webuiapi.ReActor(
            img=Image.open(source_file),
            enable=True,
            face_restorer_name='CodeFormer',
            face_restorer_visibility=1.53,
            codeFormer_weight=0.7,
        )
    ).image
    client.set_options(PERFORMANCE_WEBUI_OPTIONS)
    return result

for i in range(10):
    try:
        client.set_options(PERFORMANCE_WEBUI_OPTIONS)
        generate_performance_avatar_image(
            source_file=assets_path.joinpath(f"basic_jan.png"),
            face_file=assets_path.joinpath(f"basic_jan.png"),
            mask_file=assets_path.joinpath(f"basic_jan_mask.png")
        )
        client.set_options(PERFORMANCE_WEBUI_OPTIONS)
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

class PresetTypes(str, Enum):
    performance = "performance"
    quality = "quality"

@app.post("/generate/{avatar_type}", responses={200: {"content": {"image/png": {}}}}, response_class=Response)
async def generate(avatar_type: GenerateAvatarTypes, request: Request, file: UploadFile = File(...), preset: PresetTypes = PresetTypes.performance):
    start = time.time()
    try:
        face_file = assets_path.joinpath(f"{avatar_type.value}.png")
        mask_file = assets_path.joinpath(f"{avatar_type.value}_mask.png")
        source_file = BytesIO(await file.read())  # Solves cannot identify image file <tempfile.SpooledTemporaryFile object at 0x...>

        if not face_file.exists() or not mask_file.exists():
            raise HTTPException(status_code=404, detail=f"Avatar {avatar_type.value} not found")

        if preset == PresetTypes.quality:
            result = generate_quality_avatar_image(source_file=source_file, mask_file=mask_file, face_file=face_file)
        else:
            result = generate_performance_avatar_image(source_file=source_file, mask_file=mask_file, face_file=face_file)
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
                    "preset": preset.value,
                    "processing_time": time.time() - start,
                    "$current_url": str(request.url),
                }
            )
    except Exception as e:
        posthog.capture(request.client.host, "request_error",
            {
                "error_str": str(e),
                "traceback_str": traceback.format_exc(),
                "processing_time": time.time() - start,
                "preset": preset.value,
                "$current_url": str(request.url),
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
    start = time.time()
    try:
        background = Image.open(assets_path.joinpath("space_suit_overlay_back.png"))
        foreground = Image.open(assets_path.joinpath("space_suit_overlay_front.png"))
        source_file = Image.open(BytesIO(await file.read())).convert('RGB')  # Solves cannot identify image file <tempfile.SpooledTemporaryFile object at 0x...>
        if avatar_type.value == "me":
            source_file.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
            result = ImageOps.expand(source_file, border=20, fill=0)

            try:
                preds = fd_model.predict(np.array(result), det_threshold=.6, iou_threshold=.4)
            except RuntimeError:
                raise HTTPException(status_code=422, detail="Incorrect image for prediction")

            face = next(iter(sorted(
                [{'bbox': bbox, 'score': pred['scores'][i]} for pred in preds for i, bbox in enumerate(pred['boxes'])],
                key=lambda item: item["score"],
                reverse=True,
            )), None)
            if face is None:
                raise HTTPException(status_code=422, detail="No face found in the uploaded image!")
            result = result.crop(face['bbox']).resize((300, 400))
            background.paste(result, (360, 250))
            result = Image.alpha_composite(background, foreground)
        else:
            background.paste(source_file, (0, 0))
            result = Image.alpha_composite(background, foreground)

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
                    "$current_url": str(request.url),
                }
            )
    except Exception as e:
        posthog.capture(request.client.host, "request_error",
            {
                "error_str": str(e),
                "traceback_str": traceback.format_exc(),
                "processing_time": time.time() - start,
                "$current_url": str(request.url),
            }
        )
        raise

@app.post("/upscale/{factor}", responses={200: {"content": {"image/png": {}}}}, response_class=Response)
async def finalize(factor: float, request: Request, file: UploadFile = File(...)):
    start = time.time()
    try:
        result = client.extra_single_image(
            image=Image.open(file.file),
            upscaler_1=webuiapi.Upscaler.ESRGAN_4x,
            upscaling_resize=factor
        ).image
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
                    "result_size": len(content),
                    "processing_time": time.time() - start,
                    "$current_url": str(request.url),
                }
            )
    except Exception as e:
        posthog.capture(request.client.host, "request_error",
            {
                "error_str": str(e),
                "traceback_str": traceback.format_exc(),
                "processing_time": time.time() - start,
                "$current_url": str(request.url),
            }
        )
        raise
