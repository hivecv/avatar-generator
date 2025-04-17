import os
import time
import traceback
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
            return lambda *args, **kwargs: None
    posthog = FakeHog()

app = FastAPI()
client = webuiapi.WebUIApi(sampler='Euler', steps=20, scheduler='SGM Uniform')
assets_path = Path(__file__).parent / "assets"

def generate_avatar_image(source_file, mask_file, face_file):
    return client.img2img(
        images=[Image.open(face_file)],
        mask_image=Image.open(mask_file),
        inpainting_fill=1,
        prompt='Extreme details, high resolution, best quality, portrait warm light',
        sampler_name='SGM Uniform',
        scheduler='Euler',
        steps=10,
        seed=88888545,
        image_cfg_scale=1.5,
        cfg_scale=3.5,
        denoising_strength=0.83,
        # resize_mode=2,
        width=512,
        height=512,
        reactor=webuiapi.ReActor(
            img=Image.open(source_file),
            enable=True,
            face_restorer_name='CodeFormer',
            face_restorer_visibility=1,
            # swap_in_source=True,
            # swap_in_generated=False,
            codeFormer_weight=0.8,
            # gender_source=2 ,
            # gender_target=2,
        )
    ).image

for i in range(5):
    try:
        client.set_options({"forge_additional_modules": [
            '/stable-diffusion-webui/models/text_encoder/clip_l.safetensors',
            '/stable-diffusion-webui/models/text_encoder/clip_g.safetensors',
            '/stable-diffusion-webui/models/text_encoder/tx5xxl_fp16.safetensors'
        ]})
        generate_avatar_image(
            source_file=assets_path.joinpath(f"basic_jan.png"),
            face_file=assets_path.joinpath(f"basic_jan.png"),
            mask_file=assets_path.joinpath(f"basic_jan_mask.png")
        )
        break
    except Exception as e:
        print("Failure in generation, retrying...")
        time.sleep(1)
        if i == 4:
            raise

@app.post("/generate/{avatar_type}", responses={200: {"content": {"image/png": {}}}}, response_class=Response)
async def generate(avatar_type: str, request: Request, file: UploadFile = File(...)):
    posthog.capture(request.client.host, '$pageview', {'$current_url': str(request.url)})
    start = time.time()
    try:
        face_file = assets_path.joinpath(f"{avatar_type}.png")
        mask_file = assets_path.joinpath(f"{avatar_type}_mask.png")

        if not face_file.exists() or not mask_file.exists():
            raise HTTPException(status_code=404, detail=f"Avatar {avatar_type} not found")

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

@app.post("/finalize/{avatar_type}", responses={200: {"content": {"image/png": {}}}}, response_class=Response)
async def finalize(avatar_type: str, request: Request, file: UploadFile = File(...)):
    posthog.capture(request.client.host, '$pageview', {'$current_url': str(request.url)})
    start = time.time()
    try:
        background = Image.open(assets_path.joinpath("space_suit_overlay_back.png"))
        foreground = Image.open(assets_path.joinpath("space_suit_overlay_front.png"))
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
