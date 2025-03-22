import numpy as np
import matplotlib.pyplot as plt
from math import exp
from random import randrange, random, choice
import scipy.interpolate as interp

class IsingLattice:
    def __init__(self, size, temperature, field=0, init_mode='cold'):
        """
        Initialize the 2D Ising lattice simulation.
        
        :param size: Dimension of the square lattice.
        :param temperature: Simulation temperature.
        :param field: External magnetic field strength.
        :param init_mode: 'hot' for random (high-T) initialization or 'cold' for uniform (low-T) start.
        """
        self.size = size
        self.temperature = temperature
        self.field = field
        self.total_mag = 0.0
        self.total_energy = 0.0

        if init_mode.lower() == 'hot':
            self.spins = self._init_random()
        else:
            self.spins = self._init_uniform()

        self._compute_total_magnetization()
        self._compute_total_energy()

        self.mag_history = []  # Record of magnetization over time
        self.energy_history = []  # Record of energy over time

    def _init_uniform(self):
        """Initialize all spins to +1 (cold start)."""
        return np.ones((self.size, self.size), dtype=int)

    def _init_random(self):
        """Initialize spins randomly to +1 or -1 (hot start)."""
        lattice = np.zeros((self.size, self.size), dtype=int)
        for i in range(self.size):
            for j in range(self.size):
                lattice[i, j] = choice([1, -1])
        return lattice

    def _local_energy(self, i, j):
        """
        Compute the energy contribution from the spin at (i, j) 
        considering its four nearest neighbours with periodic boundaries.
        """
        right = self.spins[(i + 1) % self.size, j]
        left = self.spins[(i - 1 + self.size) % self.size, j]
        up = self.spins[i, (j + 1) % self.size]
        down = self.spins[i, (j - 1 + self.size) % self.size]
        neighbor_sum = right + left + up + down

        energy = -self.spins[i, j] * neighbor_sum
        if self.field:
            energy += self.spins[i, j] * self.field
        return energy

    def _compute_total_energy(self):
        """Sum over all lattice sites to obtain the total energy."""
        E_sum = 0
        for i in range(self.size):
            for j in range(self.size):
                E_sum += self._local_energy(i, j)
        self.total_energy = E_sum

    def _compute_total_magnetization(self):
        """Calculate the net magnetization of the entire lattice."""
        self.total_mag = np.sum(self.spins)

    def metropolis_step(self, steps):
        """
        Execute the Metropolis algorithm over a number of steps.
        
        :param steps: Total number of random spin flip attempts.
        """
        for _ in range(steps):
            i = randrange(self.size)
            j = randrange(self.size)

            dE = -2 * self._local_energy(i, j)

            # Accept the flip if it reduces energy or with Boltzmann probability if not
            if dE <= 0 or random() < exp(-dE / self.temperature):
                self.spins[i, j] *= -1
                self.total_mag += 2 * self.spins[i, j]
                self.total_energy += dE
                self.mag_history.append(self.total_mag)
                self.energy_history.append(self.total_energy)

class IsingPlotter:
    def __init__(self, size=10, field=0, init_mode='cold', temp_min=1, temp_max=5, temp_step=0.1, steps=50000, fixed_temp=1):
        """
        Set up the plotting tool for analyzing the Ising model simulation.
        
        :param size: Lattice dimension.
        :param field: External magnetic field strength.
        :param init_mode: Initialization mode ('cold' or 'hot').
        :param temp_min: Starting temperature for plotting.
        :param temp_max: Ending temperature for plotting.
        :param temp_step: Temperature increment.
        :param steps: Number of Metropolis steps per simulation.
        :param fixed_temp: Temperature for fixed-temperature plots.
        """
        self.size = size
        self.field = field
        self.init_mode = init_mode
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.temp_step = temp_step
        self.steps = steps
        self.fixed_temp = fixed_temp
        self.title = f"Atoms: {self.size}, Steps: {self.steps}, Field: {self.field},\nInit: {self.init_mode}, ΔT: {self.temp_step}"

    def normalize(self, data_list):
        """Normalize a list of values by the total number of spins."""
        return [val / (self.size ** 2) for val in data_list]

    def compute_specific_heat(self, energy_data, temperature):
        """Calculate specific heat from energy fluctuations."""
        avg_E = np.average(energy_data)
        avg_E2 = np.average([e**2 for e in energy_data])
        return (1 / (self.size ** 2)) * (1 / temperature**2) * (avg_E2 - avg_E**2)

    def compute_magnetic_susceptibility(self, mag_data, temperature):
        """Calculate magnetic susceptibility from magnetization fluctuations."""
        avg_M = np.average(mag_data)
        avg_M2 = np.average([m**2 for m in mag_data])
        return (1 / (self.size ** 2)) * (1 / temperature) * (avg_M2 - avg_M**2)

    def show_lattice(self, lattice_array):
        """Display the 2D lattice configuration as a heatmap."""
        plt.imshow(lattice_array, cmap='gray')
        plt.title(self.title)

    def basic_scatter(self, x_vals, y_vals, xlabel, ylabel):
        """Create a scatter plot with labels."""
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(self.title)
        plt.scatter(x_vals, y_vals)

    def interp_plot(self, x_vals, y_vals, xlabel, ylabel):
        """
        Plot both the data points and an interpolated curve using the provided x-values.
        
        :param x_vals: x-axis data.
        :param y_vals: y-axis data.
        :param xlabel: Label for x-axis.
        :param ylabel: Label for y-axis.
        """
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        # Use the provided x_vals for interpolation.
        interpolated = interp.interp1d(x_vals, y_vals)
        plt.plot(x_vals, y_vals, 'o', label="Data")
        plt.plot(x_vals, interpolated(x_vals), '-', label="Fit")
        plt.legend()

    def display(self):
        """Render the plot."""
        plt.show()

    def plot_magnetization(self):
        """Generate a plot of the average magnetization per atom vs. temperature."""
        temps = np.arange(self.temp_min, self.temp_max, self.temp_step)
        mag_vals = []
        for T in temps:
            sim = IsingLattice(self.size, T, field=self.field, init_mode=self.init_mode)
            sim.metropolis_step(self.steps)
            mag_vals.append(abs(np.average(sim.mag_history)))
        norm_mag = self.normalize(mag_vals)
        self.xlabel = "Temperature (kB/J)"
        self.ylabel = "Magnetization per atom"
        self.basic_scatter(temps, norm_mag, self.xlabel, self.ylabel)

    def plot_energy(self):
        """Generate a plot of the average energy per atom vs. temperature."""
        temps = np.arange(self.temp_min, self.temp_max, self.temp_step)
        energy_vals = []
        for T in temps:
            sim = IsingLattice(self.size, T, field=self.field, init_mode=self.init_mode)
            sim.metropolis_step(self.steps)
            energy_vals.append(np.average(sim.energy_history))
        norm_energy = self.normalize(energy_vals)
        self.xlabel = "Temperature (kB/J)"
        self.ylabel = "Energy per atom"
        self.basic_scatter(temps, norm_energy, self.xlabel, self.ylabel)

    def plot_specific_heat(self):
        """Plot the specific heat capacity per atom vs. temperature."""
        temps = np.arange(self.temp_min, self.temp_max, self.temp_step)
        heat_vals = []
        for T in temps:
            sim = IsingLattice(self.size, T, field=self.field, init_mode=self.init_mode)
            sim.metropolis_step(self.steps)
            heat_vals.append(self.compute_specific_heat(sim.energy_history, T))
        norm_heat = self.normalize(heat_vals)
        self.xlabel = "Temperature (kB/J)"
        self.ylabel = "Specific heat per atom"
        self.basic_scatter(temps, norm_heat, self.xlabel, self.ylabel)

    def plot_magnetic_susceptibility(self):
        """Plot the magnetic susceptibility per atom vs. temperature."""
        temps = np.arange(self.temp_min, self.temp_max, self.temp_step)
        susc_vals = []
        for T in temps:
            sim = IsingLattice(self.size, T, field=self.field, init_mode=self.init_mode)
            sim.metropolis_step(self.steps)
            susc_vals.append(self.compute_magnetic_susceptibility(sim.mag_history, T))
        norm_susc = self.normalize(susc_vals)
        self.xlabel = "Temperature (kB/J)"
        self.ylabel = "Magnetic susceptibility per atom"
        self.basic_scatter(temps, norm_susc, self.xlabel, self.ylabel)

    def show_spin_configuration(self):
        """Display the spin lattice after equilibration at a fixed temperature."""
        self.title = f"2D Spin Lattice at T = {self.fixed_temp}, Size: {self.size}, Init: {self.init_mode}, Field: {self.field}"
        sim = IsingLattice(self.size, self.fixed_temp, field=self.field, init_mode=self.init_mode)
        sim.metropolis_step(self.steps)
        self.spin_config = sim.spins
        self.show_lattice(self.spin_config)

    def plot_fixed_temp_magnetization(self, fixed_temp):
        """
        Plot magnetization versus magnetic field at a fixed temperature.
        """
        self.title = f"Magnetization vs Field at T = {fixed_temp}, Size: {self.size}, Init: {self.init_mode}"
        self.xlabel = "Magnetic Field"
        self.ylabel = "Magnetization"
        field_vals1 = np.arange(self.temp_min, self.temp_max, self.temp_step)
        mag_data1 = []
        for B in field_vals1:
            sim = IsingLattice(self.size, fixed_temp, field=B, init_mode=self.init_mode)
            sim.metropolis_step(self.steps)
            mag_data1.append(np.average(sim.mag_history))
        norm_data1 = self.normalize(mag_data1)[::-1]
        self.interp_plot(field_vals1, norm_data1, self.xlabel, self.ylabel)
        
        plt.figure()  # New figure for second segment
        
        field_vals2 = np.arange(self.temp_max, 2*self.temp_max, self.temp_step)
        mag_data2 = []
        for B in field_vals2:
            sim = IsingLattice(self.size, fixed_temp, field=B, init_mode=self.init_mode)
            sim.metropolis_step(self.steps)
            mag_data2.append(np.average(sim.mag_history))
        norm_data2 = self.normalize(mag_data2)
        self.interp_plot(field_vals2, norm_data2, self.xlabel, self.ylabel)

    def plot_fixed_temp_energy(self, fixed_temp):
        """
        Plot the interaction energy per atom as a function of magnetic field at a fixed temperature.
        """
        self.title = f"Energy vs Field at T = {fixed_temp}, Size: {self.size}, Init: {self.init_mode}"
        self.xlabel = "Steps"
        self.ylabel = "Energy per atom"
        for B in np.arange(self.temp_min, self.temp_max, self.temp_step):
            sim = IsingLattice(self.size, fixed_temp, field=B, init_mode=self.init_mode)
            sim.metropolis_step(self.steps)
            energy_series = sim.energy_history[::-1]
            norm_energy_series = self.normalize(energy_series)
            self.interp_plot(np.arange(self.temp_min, self.temp_max, self.temp_step), norm_energy_series, self.xlabel, self.ylabel)
            plt.figure()

    def plot_retained_magnetization(self, fixed_temp):
        """
        Plot the magnetization retention for a fixed temperature by varying the magnetic field.
        """
        self.title = f"Retained Magnetization at T = {fixed_temp}, Size: {self.size}, Init: {self.init_mode}"
        self.xlabel = "Magnetic Field"
        self.ylabel = "Magnetization"
        mag_retention = []
        for B in np.arange(self.temp_min, self.temp_max / 2, self.temp_step):
            sim = IsingLattice(self.size, fixed_temp, field=B, init_mode=self.init_mode)
            sim.metropolis_step(self.steps)
            mag_retention.append(np.average(sim.mag_history))
        for B in np.arange(self.temp_max / 2, self.temp_max, self.temp_step):
            sim = IsingLattice(self.size, fixed_temp, field=0, init_mode=self.init_mode)
            sim.metropolis_step(self.steps)
            mag_retention.append(np.average(sim.mag_history))
        norm_retention = self.normalize(mag_retention)[::-1]
        plt.plot(norm_retention)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.title)

import matplotlib.pyplot as plt

def main():
    # 1) Plot specific heat for a 10x10 lattice with a cold start
    plt.figure()  # Open a new figure
    plotter_cold = IsingPlotter(size=10, init_mode='cold', temp_step=0.1)
    plotter_cold.plot_specific_heat()

    # 2) Plot magnetization for a 15x15 lattice with a hot start
    plt.figure()  # Open a new figure
    plotter_hot = IsingPlotter(size=15, init_mode='hot', temp_step=0.1)
    plotter_hot.plot_magnetization()

    # 3) Display a 150x150 spin configuration equilibrated at T=2
    plt.figure()  # Open a new figure
    plotter_eq = IsingPlotter(size=150, init_mode='hot', fixed_temp=2)
    plotter_eq.show_spin_configuration()

    # 4) Display a 100x100 lattice with an external field of 1 and 60000 steps
    plt.figure()  # Open a new figure
    plotter_ex = IsingPlotter(size=100, field=1, init_mode='hot', steps=60000)
    plotter_ex.show_spin_configuration()

    # Finally, display all figures
    plt.show()

if __name__ == '__main__':
    main()


