"""Semi-empirical (DATCOM-style) aerodynamic coefficient generation from geometry.

Modules (each documents its equations, references and validity range):

``geometry``   body of revolution / fin planform / landing-leg geometry
``body``       body normal force & pitching moment, 0-180 deg (Allen & Perkins
               viscous crossflow as formulated by Jorgensen, NASA TN D-6996)
``nose``       nose wave drag (Linnell-Bailey cones, Rossow ogives, Taylor-Maccoll)
``friction``   compressible turbulent skin friction (van Driest II, Eckert T*)
``base``       base drag power-off / power-on, flat-face (engine-first) forebody drag
``fins``       Barrowman / Ackeret fins with body interference
``gridfins``   grid-fin normal-force and drag Mach multipliers (transonic choking)
``srp``        supersonic retro-propulsion axial-force reduction
``heating``    Sutton-Graves stagnation-point heating
``newtonian``  modified-Newtonian body integration (hypersonic blend)
``generate``   ``generate_database(vehicle) -> AeroDatabase``
``importers``  RASAero II, OpenRocket, Missile DATCOM and generic CSV importers

Import the submodules directly (``from plume.physics.aero_gen.generate import
generate_database``); the convenience names below are resolved lazily.
"""

from __future__ import annotations

import importlib

_LAZY = {
    "generate_database": "generate",
    "GeneratorOptions": "generate",
    "BodyGeometry": "geometry",
    "FinPlanform": "geometry",
    "LegAeroGeometry": "geometry",
    "gridfin_cn_alpha": "gridfins",
    "gridfin_cd0": "gridfins",
    "stagnation_heat_flux": "heating",
    "heat_load": "heating",
    "import_rasaero": "importers",
    "import_openrocket": "importers",
    "import_datcom": "importers",
    "import_generic_csv": "importers",
}

__all__ = sorted(_LAZY)


def __getattr__(name: str):
    if name in _LAZY:
        return getattr(importlib.import_module(f"{__name__}.{_LAZY[name]}"), name)
    raise AttributeError(name)
