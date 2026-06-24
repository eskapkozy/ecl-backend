import pandas as pd
from mlflow.genai.discovery import pipeline

from PDcomponent.pipelines.pdFeaturePipeline import PDFeaturePipeline
from pipelines.Features.cleaningPipeline import CleaningPipeline
from pipelines.Features.window_builder import WindowBuilder


class Orchestrator:

    def __init__(self,orig_path,hist_path,loan_number,model_config):

























