import contextlib
import importlib
import argparse
import os
import time
import urllib.parse
from types import SimpleNamespace
from vastai.vast import http_post, parse_query, show__instances, create__template, search__offers, create__instance, \
    show__instance

parser = argparse.ArgumentParser(description='Deploy avatar generator on VastAI')
parser.add_argument('api_key', help='VastAI API Key')
parser.add_argument('--posthog-key', type=str, help='PostHOG Project Key')
parser.add_argument('--disk-gb', type=int, default=80, help='Amount of disk space to take')
parser.add_argument('--eu-only', action="store_true", help='Deploy only on EU servers')
parser.add_argument('--reliability-perc', type=float, default=91.0, help='Minimum reliability threshold for server')
args = parser.parse_args()

template_id = None
template_hash = None
DAYS = 60 * 60 * 24
DESIRED_GPUS = ["GH200 SXM", "H100 PCIE", "H100 SXM", "H100 NVL"]

ON_START = f"""#!/bin/bash
bash /docker/entrypoint.sh

parallel --line-buffer ::: "bash /docker/start_webui.sh" "bash /start_avatar_api.sh {args.posthog_key}"
"""


EU_COUNTRIES = ["SE", "UA", "GB", "PL", "PT", "SI", "DE", "IT", "CH", "LT", "GR", "FI", "IS", "AT", "FR", "RO", "MD", "HU", "NO", "MK", "BG", "ES", "HR", "NL", "CZ", "EE"]


def post_wrapper(args, req_url, headers, json={}):
    global template_id, template_hash
    if urllib.parse.urlsplit(req_url).path == "/api/v0/template/":
        json["name"] = args.template_name
        json["desc"] = args.template_desc
        json["href"] = f"https://hub.docker.com/r/{args.image}/"
        response = http_post(args, req_url, headers, json)
        data = response.json()
        template_id = data["template"]["id"]
        template_hash = data["template"]["hash_id"]
        return response
    else:
        return http_post(args, req_url, headers, json)

def parse_wrapper(query_str, res = None, *args, **kwargs) -> dict:
    if isinstance(query_str, dict):
        return {
            **(res or {}),
            **query_str,
        }
    else:
        return parse_query(query_str, res, *args, **kwargs)

importlib.import_module('vastai').vast.http_post = post_wrapper
importlib.import_module('vastai').vast.parse_query = parse_wrapper

def show_instance_connection_details(instance):
    print("Connection details:")
    print(f"$ ssh -p {instance['ports']['22/tcp'][0]['HostPort']} root@{instance['public_ipaddr']} -L 8000:127.0.0.1:8000")
    print("Then, to use demo to connect, execute:")
    print("$ cd demo")
    print("$ python3 -m pip install -r requirements.txt")
    print("$ python3 -m streamlit run demo.py")

base_settings = SimpleNamespace(
    url="https://console.vast.ai",
    api_key=args.api_key,
    explain=False,
    retry=1,
    raw=True,
    no_default=True,
)

def create_instance():
    search_offers_settings = SimpleNamespace(
        **base_settings.__dict__,
        query={
            "verified": {"eq": True},
            "external": {"eq": False},
            "rentable": {"eq": True},
            "rented": {"eq": False},
            "disk_space": {"gte": args.disk_gb},
            "reliability2": {"gte": args.reliability_perc / 100},
            "duration": {"gte": 1 * DAYS},
            "num_gpus": {"gte": 1, "lte": 1},
            "sort_option": {"0": ["dph_total", "asc"], "1": ["total_flops", "asc"]},
            "gpu_name": {"in": DESIRED_GPUS},
            "gpu_totalram": {"gte": 16384},
            "direct_port_count": {"gte":2},
            **({"geolocation": {"in": EU_COUNTRIES}} if args.eu_only else {})
        },
        order="dph_total,total_flops",
        type="ask",
        limit=10,
        storage=args.disk_gb,
        disable_bundling=False,
        new=False,
    )
    offers = search__offers(search_offers_settings)

    def show_offer(offer):
        print(
            f"{offer['id']} \t| {int(offer['score'])} \t| {offer['geolocation'].ljust(20)} \t| {offer['gpu_name'].ljust(10)} \t| {offer['cuda_max_good']} \t| {offer['inet_up']} / {offer['inet_down']} \t| {int(offer['reliability'] * 1000) / 10}% \t| {offer['discounted_dph_total']}")

    print("=== Best 10 offers ===")
    for item in offers:
        show_offer(item)
    print("======================")

    print("=== Selected offer ===")
    selected_offer = offers[0]
    show_offer(selected_offer)
    print("======================")

    create_instance_settings = SimpleNamespace(
        **base_settings.__dict__,
        id=selected_offer['id'],
        template_hash=template_hash,
        disk=args.disk_gb,
        onstart=None,
        entrypoint=None,
        onstart_cmd=ON_START,
        image=None,
        env=None,
        bid_price=None,
        label=None,
        extra=None,
        login=None,
        python_utf8=None,
        lang_utf8=None,
        jupyter_lab=None,
        jupyter_dir=None,
        force=None,
        cancel_unavail=None,
        args=None,
    )

    new_contract_id = create__instance(create_instance_settings).json()['new_contract']
    print("NEW CONTRACT_ID: ", new_contract_id)
    return new_contract_id

show_instances_settings = SimpleNamespace(
    **base_settings.__dict__,
    quiet=False,
)

create_template_settings = SimpleNamespace(
    **base_settings.__dict__,
    jupyter=False,
    jupyter_lab=False,
    jupyter_dir=None,
    direct=True,
    ssh=True,
    login=None,
    search_params=None,
    image="hivecv/avatar-generator",
    image_tag="main",
    env=None,
    onstart_cmd=ON_START,
    disk_space=args.disk_gb,
    template_name="Avatar Generator",
    template_desc="Generate avatars based on your face",
)

with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
    create__template(create_template_settings)

print("TEMPLATE ID: ", template_id)
print("TEMPLATE HASH: ", template_hash)

for instance in show__instances(show_instances_settings):
    if instance["gpu_name"] in DESIRED_GPUS:
        print("Existing instance found")
        instance_id = instance['id']
        break
else:
    print("Existing instance not found")
    instance_id = create_instance()

show_instance_settings = SimpleNamespace(
    **base_settings.__dict__,
    id=instance_id,
)

ssh_port = None
while ssh_port is None:
    instance = show__instance(show_instance_settings)
    if '22/tcp' in instance.get('ports', {}):
        ssh_port = instance['ssh_port']
    else:
        print("Waiting for SSH port...")
        time.sleep(10)

show_instance_connection_details(instance)


