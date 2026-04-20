This is a placeholder for the trained model weights file (model.pth).

To generate a real dummy weights file, run:

    python -c "
    import torch
    from model.model_loader import DummyDRModel
    model = DummyDRModel()
    torch.save(model.state_dict(), 'saved_models/model.pth')
    print('Saved dummy model.pth')
    "

To use a real trained model:
    1. Train your model and save: torch.save(model.state_dict(), 'saved_models/model.pth')
    2. Make sure the model architecture in model/model_loader.py matches your trained model.
    3. The app will automatically load the weights on startup.

NOTE: Do NOT commit large .pth files to Git. Use Git LFS or DVC for model versioning.
