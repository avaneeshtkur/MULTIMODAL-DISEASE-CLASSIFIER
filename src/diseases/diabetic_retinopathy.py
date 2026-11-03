"""
diabetic_retinopathy.py — DR screening on retinal fundus photographs.

Binary only: the output is the model's two classes (e.g. No_DR vs DR, or
non-referable vs referable — whatever metadata.class_labels says). Five-grade
severity is never inferred from a binary model.

Input domain: the bundled DR dataset consists of 224 px *Gaussian-filtered*
("Ben Graham") fundus images with a grey background. If metadata declares
"input_domain": "ben_graham_filtered", raw photographs (black surround) are
out of distribution and the result is INCONCLUSIVE rather than forced.
"""
from src.diseases.base import SpecialistModule

RAW_FUNDUS_DARK_CORNER_FRAC = 0.5


class DRModule(SpecialistModule):
    key = "dr"
    modality = "fundus"
    default_labels = ("No_DR", "DR")

    def input_domain_issue(self, image, metrics):
        domain = self.handle.metadata.get("input_domain")
        if domain == "ben_graham_filtered" and metrics.get("dark_corner_frac", 0) > RAW_FUNDUS_DARK_CORNER_FRAC:
            return ("This DR model was trained on Gaussian-filtered (Ben Graham) fundus images. "
                    "The uploaded image looks like an unfiltered photograph (dark surround), "
                    "which is outside the model's training distribution.")
        return None
