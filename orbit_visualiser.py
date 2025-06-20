import numpy as np
import matplotlib.pyplot as plt
from O_elements_to_cartesian import kepler_to_cartesian
# from orbdtools import KeprvTrans

class LunarOrbitVisualizer:
    def __init__(self):
        self.orbits = []
        self.points = []
        self.R_MOON = 1737  # Moon radius in km

    def kepler_to_position(self, a, e, i, raan, arg_periapsis, true_anomaly_deg):
        """Convert Keplerian elements to Cartesian coordinates."""
        mu_moon = 4902.800066

        # My own implementation
        tvec = kepler_to_cartesian(a, e, i, raan, arg_periapsis, true_anomaly_deg, mu_moon)[0]

        # KeprvTrans implementation (same result)
        # tvec = KeprvTrans.coe2rv(np.array([a, e, i, raan, arg_periapsis, true_anomaly_deg]), mu_moon)[:3]
        
        return tvec

    def orbital_elements_to_position(self, a, e, i, raan, arg_periapsis, true_anomaly_deg):
        """
        Currently unused. Use kepler_to_position instead which is the same but keeps code consistent. 
        """
        i = np.radians(i)
        raan = np.radians(raan)
        arg_periapsis = np.radians(arg_periapsis)
        true_anomaly = np.radians(true_anomaly_deg)

        r = a * (1 - e**2) / (1 + e * np.cos(true_anomaly))
        x_p = r * np.cos(true_anomaly)
        y_p = r * np.sin(true_anomaly)
        z_p = 0

        cos_raan = np.cos(raan)
        sin_raan = np.sin(raan)
        cos_i = np.cos(i)
        sin_i = np.sin(i)
        cos_argp = np.cos(arg_periapsis)
        sin_argp = np.sin(arg_periapsis)

        R = np.array([
            [cos_raan * cos_argp - sin_raan * sin_argp * cos_i,
             -cos_raan * sin_argp - sin_raan * cos_argp * cos_i,
             sin_raan * sin_i],
            [sin_raan * cos_argp + cos_raan * sin_argp * cos_i,
             -sin_raan * sin_argp + cos_raan * cos_argp * cos_i,
             -cos_raan * sin_i],
            [sin_argp * sin_i,
             cos_argp * sin_i,
             cos_i]
        ])

        return np.dot(R, np.array([x_p, y_p, z_p]))

    def add_orbit(self, a, e, i, raan, arg_periapsis, true_anomalies, label=None):
        points = np.array([
            self.kepler_to_position(a, e, i, raan, arg_periapsis, nu)
            for nu in true_anomalies
        ])
        self.orbits.append({'points': points, 'label': label})

    def add_points(self, point_list, label=None):
        """
        Add a set of 3D points to be plotted.
        :param point_list: list of np.array([x, y, z])
        :param label: optional label for the point set
        """
        points = np.array(point_list)
        self.points.append({'points': points, 'label': label})

    def plot(self):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        all_plot_points = []

        for orbit in self.orbits:
            ax.plot(orbit['points'][:, 0], orbit['points'][:, 1], orbit['points'][:, 2],
                    label=orbit['label'] or 'Orbit')
            
            # This will plot all orbit points in red. We want the orbit drawn as just a line and then specific points plotted.
            # Uncomment the next 2 lines to plot all points in red
            # ax.scatter(orbit['points'][:, 0], orbit['points'][:, 1], orbit['points'][:, 2],
            #            s=5, color='red')
            all_plot_points.append(orbit['points'])

        for idx, point_set in enumerate(self.points):
            pts = point_set['points']
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=20, label=point_set['label'] or f'Point Set {idx+1}', color='red')
            all_plot_points.append(pts)

        # Moon sphere
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x = self.R_MOON * np.outer(np.cos(u), np.sin(v))
        y = self.R_MOON * np.outer(np.sin(u), np.sin(v))
        z = self.R_MOON * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(x, y, z, color='gray', alpha=0.5)

        # Equal aspect ratio setup
        if all_plot_points:
            all_points = np.vstack(all_plot_points)
            max_range = np.array([
                all_points[:, 0].ptp(),
                all_points[:, 1].ptp(),
                all_points[:, 2].ptp()
            ]).max() / 2.0

            mid_x = all_points[:, 0].mean()
            mid_y = all_points[:, 1].mean()
            mid_z = all_points[:, 2].mean()

            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.set_box_aspect([1, 1, 1])
        ax.set_title("Lunar Orbit Visualization with Points")
        ax.set_xlabel("X (km)")
        ax.set_ylabel("Y (km)")
        ax.set_zlabel("Z (km)")
        ax.legend()
        plt.tight_layout()
        plt.show()

visualizer = LunarOrbitVisualizer()
true_anomalies = np.linspace(0, 360, 100)
true_anomalies_calc = np.linspace(0, 230, 10)

true_anomalies_calc = np.array([0, 25.68732493, 51.2271774, 76.76702987, 102.30688234,
                                127.84673481, 153.38658728, 178.77896728,
                                204.31881975, 229.85867222])

visualizer.add_orbit(
    a=1837, e=0.01, i=90, raan=45, arg_periapsis=0,
    true_anomalies=true_anomalies,
    label='Apollo-style Orbit'
)
visualizer.plot()
