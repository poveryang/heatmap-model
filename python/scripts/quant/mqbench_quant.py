from collections import OrderedDict
import sys
from pathlib import Path

import torch
import tqdm
from mqbench.convert_deploy import convert_deploy  # remove quant nodes for deploy
from mqbench.prepare_by_platform import BackendType  # contain various Backend, like TensorRT, NNIE, etc.
from mqbench.prepare_by_platform import prepare_by_platform  # add quant nodes for specific Backend
from mqbench.utils.state import enable_calibration  # turn on calibration algorithm, determine scale, zero_point, etc.
from mqbench.utils.state import enable_quantization  # turn on actually quantization, like FP32 -> INT8

PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from hmap.dataset import HMapDataModule
from hmap.model import FocalLoss, UNet


class UNetTrans(UNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        x = super().forward(x)
        x = x.permute(0, 2, 3, 1)  # chw -> hwc
        return x


def load_model(model_path):
    model_fp32 = UNetTrans(in_channels=1, n_classes=3, inc_channels=16, interpolation='nearest')
    pl_state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    state_dict = OrderedDict()
    for k, v in pl_state_dict['state_dict'].items():
        k = k.replace('generator.', '')
        state_dict[k] = v
    model_fp32.load_state_dict(state_dict)
    return model_fp32


def post_train_quant(model_path):
    # 1. Load model and dataloader
    model = load_model(model_path)
    model.eval()
    datamodule = HMapDataModule(root_dir='/home/yjunj/Datasets/barcode',
                                input_size=[400, 640],
                                batch_size=4)
    datamodule.setup()
    dataloader = datamodule.train_dataloader()

    # 2. Set Backend
    backend = BackendType.Tengine_u8

    # 3. Prepare model for Backend
    model = prepare_by_platform(model, backend)  # trace model and add quant nodes for model on Tengine Backend

    # 4. Calibration
    enable_calibration(model)  # turn on calibration, ready for gathering data
    for batch_idx, (data, target) in enumerate(dataloader):
        # run the training step
        _ = model(data)

        print(batch_idx, end=' ', flush=True)
        if batch_idx >= 200:
            break

    # 5. Quantization
    enable_quantization(model)  # turn on actually quantization, ready for simulating Backend inference
    for batch_idx, (data, target) in enumerate(dataloader):
        # run the forward step
        _ = model(data)

    # 6. Convert to deploy
    input_shape = {'data': [1, 1, 400, 640]}
    convert_deploy(model, backend, input_shape)  # remove quant nodes, ready for deploying to real-world hardware


def quant_aware_train(model_path, num_epochs=10):
    # 1. Load model and dataloader
    model_fp32 = load_model(model_path)
    model_fp32.eval()
    datamodule = HMapDataModule(root_dir='/home/yjunj/Datasets/barcode',
                                input_size=[400, 640],
                                batch_size=8)
    datamodule.setup()
    dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()

    # 2. Set Backend
    backend = BackendType.Tengine_u8

    # 3. Prepare model for Backend
    model_prepared = prepare_by_platform(model_fp32, backend)  # trace model and add quant nodes for model
    model_prepared.to('cuda')

    # 4. Calibration
    enable_calibration(model_prepared)  # turn on calibration, ready for gathering data
    model_prepared.eval()
    for batch_idx, (image, target) in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        _ = model_prepared(image.to('cuda'))  # run the forward step

    # 5. Quantization aware training
    enable_quantization(model_prepared)  # turn on actually quantization, ready for simulating Backend inference
    model_prepared.train()

    loss_fn = FocalLoss()
    lr = 5e-6

    for epoch in range(num_epochs):
        optimizer = torch.optim.Adam(model_prepared.parameters(), lr=lr)
        train_loss = train_one_epoch(model_prepared, dataloader, loss_fn, optimizer)
        val_loss = train_one_epoch(model_prepared, val_dataloader, loss_fn, optimizer)
        lr = lr * 0.8

        print(f'epoch: {epoch}, train loss: {train_loss}, val loss: {val_loss}')
        # 6. Convert to deploy
        input_shape = {'data': [1, 1, 400, 640]}
        # remove quant nodes, ready for deploying to real-world hardware
        convert_deploy(model_prepared, backend, input_shape, model_name=f'hmap-v2-qat-{epoch}-{val_loss:.6f}')


def train_one_epoch(model, dataloader, loss_fn, optimizer):
    losses = []
    model.train()
    for batch_idx, (image, target) in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        optimizer.zero_grad()
        output = model(image.to('cuda'))
        target = target.permute(0, 2, 3, 1)
        loss = loss_fn(output, target.to('cuda'))
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    ave_loss = sum(losses) / len(losses)
    return ave_loss


def export_model(model_path):
    model_fp32 = load_model(model_path)
    model_fp32.eval()
    datamodule = HMapDataModule(root_dir='/Users/yjunj/WorkSpace/Datasets/barcode',
                                input_size=[400, 640],
                                batch_size=1)
    datamodule.setup()
    dataloader = datamodule.train_dataloader()
    print(len(dataloader))

    backend = BackendType.Tengine_u8
    model_prepared = prepare_by_platform(model_fp32, backend)

    qat_model_path = FIXTURES_DIR / 'checkpoints' / 'hmap-v1-qat-epoch=9.ckpt'
    state_dict = torch.load(qat_model_path, map_location=torch.device('cpu'))
    model_prepared.load_state_dict(state_dict, strict=False)

    model_prepared.eval()

    _ = FocalLoss()
    enable_calibration(model_prepared)
    for batch_idx, (image, target) in tqdm.tqdm(enumerate(dataloader)):
        # run the training step
        _ = model_prepared(image)
        # loss = loss_fn(output, target)
        # tqdm.tqdm.write(f'loss: {loss.item():.6f}', nolock=True)
        if batch_idx >= 50:
            break
    #
    enable_quantization(model_prepared)
    for batch_idx, (image, target) in tqdm.tqdm(enumerate(dataloader)):
        # run the training step
        _ = model_prepared(image)
        # loss = loss_fn(output, target)
        # tqdm.tqdm.write(f'loss: {loss.item():.6f}', nolock=True)
        if batch_idx >= 50:
            break

    input_shape = {'data': [1, 1, 400, 640]}
    # remove quant nodes, ready for deploying to real-world hardware
    convert_deploy(model_prepared, backend, input_shape, model_name='hmap-v1-qat')


def main():
    import argparse
    parser = argparse.ArgumentParser(description='MQBench PTQ/QAT helper.')
    parser.add_argument('--ckpt', required=True, help='Path to fp32 checkpoint.')
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    quant_aware_train(args.ckpt, num_epochs=args.epochs)


if __name__ == '__main__':
    main()
