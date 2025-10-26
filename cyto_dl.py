from cyto_dl.api import CytoDLModel
import asyncio

model = CytoDLModel()
model.download_example_data()
model.load_default_experiment("segmentation", output_dir="./output", overrides=["trainer=cpu"])
model.print_config()
model.train()

# [OPTIONAL] async training
await model.train(run_async=True)
