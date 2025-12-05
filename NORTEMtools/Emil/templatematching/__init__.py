from . import io, postprocessing
import NORTEMtools.Emil.utils as _emutils
from NORTEMtools import _logger

import matplotlib.pyplot as plt
from diffpy.structure import loadStructure
from orix.crystal_map import Phase
from orix.sampling import get_sample_reduced_fundamental
from diffsims.generators.simulation_generator import SimulationGenerator
from orix.quaternion import Orientation


class MyBank:
    """
    Class to create, simulate and handle template banks in a more streamlined manner
    """

    __simulation_generator_defaults__ = {
        "precession_angle": 1.0,
        "minimum_intensity": 1e-20,
        "shape_factor_model": "linear",
        "approximate_precession": True,
        "scattering_params": "lobato",
        "accelerating_voltage": 200,
    }

    __simulation_defaults__ = {
        "with_direct_beam": False,
        "max_excitation_error": 1e-1,
        "shape_factor_width": None,
        "debye_waller_factors": None,
    }

    def __init__(
        self,
        name,
        cif_file,
        space_group,
        angular_resolution=0.5,
        reciprocal_radius=2.0,
        generator_kwargs={},
        simulation_kwargs={},
    ):
        f"""
        Create a new template bank

        :param name: Name of the bank
        :param cif_file: The cif file to use for the bank
        :param space_group: The space group of the crystal
        :param angular_resolution: The angular resolution to to use when simulating banks
        :param reciprocal_radius: The maximum reciprocal radius to simulate
        :param generator_kwargs: Keyword arguments to be passed to the generator generation. Default values are {self.__simulation_generator_defaults__}. See `diffsims.generator.simulation_generator.SimulationGenerator` for more information.
        :param simulation_kwargs: Keyword arguments to be passed to the simulation. Default values are {self.__simulation_defaults__}. See `diffsims.generator.simulation_generator.SimulationGenerator.calculate_diffraction2d()` for more information.
        """

        self.name = str(name)
        self.space_group = int(space_group)
        self.cif_file = _emutils.MyPath(cif_file)
        self.angular_resolution = float(angular_resolution)
        self.reciprocal_radius = float(reciprocal_radius)
        self.simulation_generator = None
        self.simulations = None
        self.structure = None
        self.phase = None
        self.directions = None
        self.orientations = None

        self.set_simulation_generator(**generator_kwargs)
        self.simulate(**simulation_kwargs)

    def _load_structure(self):
        self.structure = loadStructure(str(self.cif_file))

    def _set_phase(self):
        self._load_structure()
        self.phase = Phase(
            name=self.name, space_group=self.space_group, structure=self.structure
        )

    def _set_directions(self):
        self._set_phase()
        self.directions = get_sample_reduced_fundamental(
            self.angular_resolution, point_group=self.phase.point_group
        )
        self.orientations = Orientation(
            self.directions, symmetry=self.phase.point_group
        )
        self.orientations.scatter("ipf")

    def set_simulation_generator(self, *args, **kwargs):
        for param in self.__simulation_generator_defaults__:
            kwargs[param] = kwargs.get(
                param, self.__simulation_generator_defaults__[param]
            )
        self.simulation_generator = SimulationGenerator(*args, **kwargs)

    def simulate(self, *args, **kwargs):
        self._set_directions()
        for param in self.__simulation_defaults__:
            kwargs[param] = kwargs.get(param, self.__simulation_defaults__[param])

        kwargs["phase"] = self.phase
        kwargs["rotation"] = (self.directions,)
        kwargs["reciprocal_radius"] = self.reciprocal_radius

        print(
            f"I will simulate patterns of {self.name} up to {self.reciprocal_radius:.3f} Å^-1"
        )
        self.simulations = self.simulation_generator.calculate_diffraction2d(
            *args, **kwargs
        )
        print(
            f"Simulated {self.simulations.current_size} patterns for {self.name} template bank"
        )

    def plot_orientations(self):
        plt.figure()
        self.orientations.scatter("ipf")

    def plot_simulations(self):
        plt.figure()
        self.simulations.plot(interactive=True)

    def save_templates(self, path):
        path = _emutils.MyPath(path)
        if path.is_dir():
            path = path / self.name
            path = path.with_suffix(".pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        io.save_template(self.simulations, path)


__all__ = ["io", "postprocessing", "MyBank"]
