FROM alpine/git:2.36.2 as download

COPY clone.sh /clone.sh

RUN . /clone.sh stable-diffusion-webui-assets https://github.com/AUTOMATIC1111/stable-diffusion-webui-assets.git 6f7db241d2f8ba7457bac5ca9753331f0c266917

RUN . /clone.sh stable-diffusion-stability-ai https://github.com/Stability-AI/stablediffusion.git cf1d67a6fd5ea1aa600c4df58e5b47da45f6bdbf \
  && rm -rf assets data/**/*.png data/**/*.jpg data/**/*.gif

RUN . /clone.sh BLIP https://github.com/salesforce/BLIP.git 48211a1594f1321b00f14c9f7a5b4813144b2fb9
RUN . /clone.sh k-diffusion https://github.com/crowsonkb/k-diffusion.git ab527a9a6d347f364e3d185ba6d714e22d80cb3c
RUN . /clone.sh clip-interrogator https://github.com/pharmapsychotic/clip-interrogator 2cf03aaf6e704197fd0dae7c7f96aa59cf1b11c9
RUN . /clone.sh generative-models https://github.com/Stability-AI/generative-models 45c443b316737a4ab6e40413d7794a7f5657c19f
RUN . /clone.sh stable-diffusion-webui-assets https://github.com/AUTOMATIC1111/stable-diffusion-webui-assets 6f7db241d2f8ba7457bac5ca9753331f0c266917


FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive PIP_PREFER_BINARY=1

RUN --mount=type=cache,target=/var/cache/apt \
  apt-get update && \
  # we need those
  apt-get install -y fonts-dejavu-core rsync git jq moreutils aria2 parallel \
  # extensions needs those
  ffmpeg libglfw3-dev libgles2-mesa-dev pkg-config libcairo2 libcairo2-dev build-essential


WORKDIR /
RUN --mount=type=cache,target=/root/.cache/pip \
  git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git && \
  cd stable-diffusion-webui  && \
  # Forge
  git remote add forge https://github.com/lllyasviel/stable-diffusion-webui-forge &&\
  git fetch forge && \
  git checkout -b using_forge forge/main && \
  git pull && \
  pip install -r requirements_versions.txt


ENV ROOT=/stable-diffusion-webui

COPY --from=download /repositories/ ${ROOT}/repositories/
RUN mkdir ${ROOT}/interrogate && cp ${ROOT}/repositories/clip-interrogator/clip_interrogator/data/* ${ROOT}/interrogate

RUN --mount=type=cache,target=/root/.cache/pip \
   pip uninstall -y typing_extensions huggingface-guess gradio && \
   pip install typing_extensions==4.11.0 huggingface-guess==0.1.0 gradio==5.25.1 "cython<3.0.0" wheel

COPY assets /assets/

RUN --mount=type=cache,target=/root/.cache/pip \
   pip install --prefer-binary --no-build-isolation "fastapi[standard]"==0.115.11 webuiapi==0.9.17 pillow==11.1.0 python-multipart==0.0.20 posthog==3.21.0 fastface

COPY avatar_api.py start_avatar_api.sh /

RUN --mount=type=cache,target=/root/.cache/pip \
  pip install pyngrok xformers==0.0.26.post1 \
  git+https://github.com/TencentARC/GFPGAN.git@8d2447a2d918f8eba5a4a01463fd48e45126a379 \
  git+https://github.com/openai/CLIP.git@d50d76daa670286dd6cacf3bcd80b5e4823fc8e1 \
  git+https://github.com/mlfoundations/open_clip.git@v2.20.0

# there seems to be a memory leak (or maybe just memory not being freed fast enough) that is fixed by this version of malloc
# maybe move this up to the dependencies list.
RUN apt-get -y install libgoogle-perftools-dev netcat && apt-get clean
ENV LD_PRELOAD=libtcmalloc.so

COPY clone.sh config.py entrypoint.sh start_webui.sh download.sh links.txt checksums.sha256 /docker/

RUN \
  # mv ${ROOT}/style.css ${ROOT}/user.css && \
  # one of the ugliest hacks I ever wrote \
  sed -i 's/in_app_dir = .*/in_app_dir = True/g' /opt/conda/lib/python3.10/site-packages/gradio/routes.py && \
  git config --global --add safe.directory '*'

WORKDIR ${ROOT}
ENV NVIDIA_VISIBLE_DEVICES=all
ENV CLI_ARGS=""
ENTRYPOINT ["/docker/entrypoint.sh"]
CMD parallel --line-buffer ::: "bash /docker/start_webui.sh" "bash /start_avatar_api.sh"