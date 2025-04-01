from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path



@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int



def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
    config = self.config.prepare_base_model
    
    create_directories([config.root_dir])

    prepare_base_model_config = PrepareBaseModelConfig(
        root_dir=Path(config.root_dir),
        base_model_path=Path(config.base_model_path),
        updated_base_model_path=Path(config.updated_base_model_path),
        params_image_size=self.params.IMAGE_SIZE,
        params_learning_rate=self.params.LEARNING_RATE,
        params_include_top=self.params.INCLUDE_TOP,
        params_weights=self.params.WEIGHTS,
        params_classes=self.params.CLASSES
    )

    return prepare_base_model_config
    


@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list



@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path
    training_data: Path
    all_params: dict
    #mlflow_uri: str
    params_image_size: list
    params_batch_size: int