import torch


def load_pl_model(model_type, model_path=None, **model_kwargs):
    """
    Load a PyTorch Lightning model.

    Args:
        model_type: the type of model
        model_path: the path of model
        **model_kwargs: the arguments of model

    Returns: a PyTorch Lightning model

    """
    # Initialize model
    pl_model = model_type(**model_kwargs)

    # Load model
    if model_path is not None:
        pl_state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        pl_model.load_state_dict(pl_state_dict['state_dict'])

    return pl_model
