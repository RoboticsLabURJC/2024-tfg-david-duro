BasicLidar = SensorParams(
    start_angle=0,
    end_angle=2 * np.pi,
    laser_angles=np.linspace(-np.radians(0.01), np.radians(0.01), 1),
    angle_resolution=0.02094,
    max_distance=20,
    noise_mu=0,
    noise_sigma=0.078,
)

