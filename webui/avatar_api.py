import os
import time
import traceback
from io import BytesIO
from pathlib import Path
from PIL import Image
import webuiapi
from fastapi import FastAPI, File, Request, Response, UploadFile, HTTPException

if 'POSTHOG_KEY' in os.environ:
    from posthog import Posthog
    posthog = Posthog(project_api_key=os.environ['POSTHOG_KEY'], host='https://eu.i.posthog.com')
else:
    class FakeHog:
        def __getattr__(self):
            return lambda *args, **kwargs: None
    posthog = FakeHog()

app = FastAPI()
client = webuiapi.WebUIApi()
assets_path = Path(__file__).parent / "assets"


@app.post("/generate/{avatar_type}", responses={200: {"content": {"image/png": {}}}}, response_class=Response)
async def generate(avatar_type: str, request: Request, file: UploadFile = File(...)):
    posthog.capture(request.client.host, '$pageview', {'$current_url': str(request.url)})
    start = time.time()
    try:
        face_file = assets_path.joinpath(f"{avatar_type}.png")
        mask_file = assets_path.joinpath(f"{avatar_type}_mask.png")

        if not face_file.exists() or not mask_file.exists():
            raise HTTPException(status_code=404, detail=f"Avatar {avatar_type} not found")

        result = client.img2img(
            images=[Image.open(face_file)],
            mask_image=Image.open(mask_file),
            inpainting_fill=1,
            prompt="realistic",
            steps=25,
            seed=-1,
            cfg_scale=7,
            denoising_strength=0.5,
            resize_mode=2,
            width=1024,
            height=1024,
            reactor=webuiapi.ReActor(
                img=Image.open(file.file),
                enable=True
            )
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
