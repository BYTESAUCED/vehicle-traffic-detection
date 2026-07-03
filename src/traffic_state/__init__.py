"""BMD-45 traffic-state package.

Vehicle detection on traffic-camera images using an RF-DETR-Nano model
fine-tuned on a BMD-45 subset, plus simple roadway-density labelling.
"""

from traffic_state.density import DensityThresholds, density_label, load_thresholds

__all__ = ["DensityThresholds", "density_label", "load_thresholds", "__version__"]

__version__ = "0.1.0"
