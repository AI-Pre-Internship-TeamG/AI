#!/usr/bin/env python3

from werkzeug.datastructures import FileStorage
import io, os, uuid, cv2, torch
from typing import Union
import numpy as np
from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config
from lama_cleaner.model.connection import saveImageToS3
from flask import request,Flask
from lama_cleaner.environ import start
from lama_cleaner.helper import (
    load_img,
    numpy_to_bytes,
    resize_max_size,
    get_image,
    get_image_ext
)
try:
    torch._C._jit_override_can_fuse_on_cpu(False)
except:
    pass

start()
BUILD_DIR = os.environ.get("LAMA_CLEANER_BUILD_DIR", "app/build")
app = Flask(__name__, static_folder=os.path.join(BUILD_DIR, "static"))

model: ModelManager = None
device = None
input_image_path: str = None

@app.route("/inpaint", methods=["POST"])
def process():
    input = request.form
    origin_image_url = input["fname"]
    slice_origin_image = origin_image_url[54:]
    origin_image = get_image(slice_origin_image)
    image, alpha_channel = load_img(origin_image)
    interpolation = cv2.INTER_CUBIC

    mask_image_url = input["mask"]
    slice_mask_image = mask_image_url[54:]
    mask_image = get_image(slice_mask_image)
    mask, _ = load_img(mask_image, gray=True)

    size_limit: Union[int, str] = input.get("sizeLimit", "1080")
    if size_limit == "Original":
        size_limit = max(image.shape)
    else:
        size_limit = int(size_limit)

    config = Config(
        ldm_steps=0,
        ldm_sampler=0,
        hd_strategy=0,
        zits_wireframe=0,
        hd_strategy_crop_margin=0,
        hd_strategy_crop_trigger_size=0,
        hd_strategy_resize_limit=0,
    )

    image = resize_max_size(image, size_limit=size_limit, interpolation=interpolation)
    mask = resize_max_size(mask, size_limit=size_limit, interpolation=interpolation)

    res_np_img = model(image, mask, config)

    if alpha_channel is not None:
        if alpha_channel.shape[:2] != res_np_img.shape[:2]:
            alpha_channel = cv2.resize(
                alpha_channel, dsize=(res_np_img.shape[1], res_np_img.shape[0])
            )
        res_np_img = np.concatenate(
            (res_np_img, alpha_channel[:, :, np.newaxis]), axis=-1
        )

    f = io.BytesIO(numpy_to_bytes(res_np_img, 'png'))
    c = str(uuid.uuid4())
    inpaint_image = FileStorage(f, c, name='file', content_type='image/png')
    inpaint_image = saveImageToS3(inpaint_image,'result')

    return inpaint_image

def main(args):
    global model
    global device
    global input_image_path

    device = torch.device(args.device)
    input_image_path = args.input

    model = ModelManager(name=args.model, device=device)

    if args.gui:
        app_width, app_height = args.gui_size
        from flaskwebgui import FlaskUI

        ui = FlaskUI(
            app, width=app_width, height=app_height, host=args.host, port=args.port
        )
        ui.run()
    else:
        app.run(host=args.host, port=args.port, debug=args.debug)
