from src.diseases.diabetic_retinopathy import DRModule
from src.diseases.oral_cancer import OralCancerModule
from src.diseases.tb import TBModule

MODULE_CLASSES = {"tb": TBModule, "dr": DRModule, "oral": OralCancerModule}


def build_modules(registry):
    return {k: cls(registry) for k, cls in MODULE_CLASSES.items()}
