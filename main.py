from __future__ import annotations

import datetime
import typing

import matplotlib.axes
import matplotlib.patches
import matplotlib.pyplot
import numpy
import numpy.random
import sklearn.mixture


class Target:
    def __init__(
        self,
        extract_features: typing.Callable[[Target], numpy.ndarray],
        time_delta: datetime.timedelta,
        acceleration_standard_deviation: float,
    ) -> None:
        self._extract_features = extract_features
        self._transition_matrix = numpy.array(
            [
                [1, 0, time_delta.total_seconds(), 0],
                [0, 1, 0, time_delta.total_seconds()],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        self._acceleration_standard_deviation = acceleration_standard_deviation
        self._process_noise_matrix = numpy.array(
            [
                [time_delta.total_seconds() ** 4 / 4, time_delta.total_seconds() ** 3 / 2, 0, 0],
                [time_delta.total_seconds() ** 3 / 2, time_delta.total_seconds() ** 2, 0, 0],
                [0, 0, time_delta.total_seconds() ** 4 / 4, time_delta.total_seconds() ** 3 / 2],
                [0, 0, time_delta.total_seconds() ** 3 / 2, time_delta.total_seconds() ** 2],
            ]
        )
        self._state = numpy.array([[0, 0, 3, 0]]).transpose()
        self._noise_generator = numpy.random.default_rng()

    def step(self) -> None:
        self._state = (
            self._transition_matrix @ self._state + self._process_noise_matrix @ self._make_noise()
        )

    def _make_noise(self) -> numpy.ndarray:
        noise = self._noise_generator.normal(
            numpy.zeros((2,)), numpy.array([self._acceleration_standard_deviation] * 2)
        )
        return numpy.array([[noise[0]] * 2 + [noise[1]] * 2]).transpose()

    @property
    def state(self) -> numpy.ndarray:
        return self._state

    @property
    def features(self) -> numpy.ndarray:
        return self._extract_features(self)


class TargetInvariantTargetFeatureExtractor:
    def __init__(
        self, features_mean: numpy.ndarray, features_covariance: numpy.ndarray, identifier: int
    ) -> None:
        self._features_mean = features_mean
        self._features_covariance = features_covariance
        self._identifier = identifier
        self._noise_generator = numpy.random.default_rng()

    def extract_features(self, target: Target) -> numpy.ndarray:
        return numpy.insert(
            numpy.expand_dims(
                self._noise_generator.multivariate_normal(
                    self._features_mean, self._features_covariance
                ),
                0,
            ).transpose(),
            0,
            self._identifier,
            0,
        )


class Clutter:
    def __init__(
        self,
        features_mean: numpy.ndarray,
        features_covariance: numpy.ndarray,
        minimum_clutter_count: int,
        maximum_clutter_count: int,
        identifier: int,
    ) -> None:
        self._features_mean = features_mean
        self._features_covariance = features_covariance
        self._minimum_clutter_count = minimum_clutter_count
        self._maximum_clutter_count = maximum_clutter_count
        self._identifier = identifier
        self._features = numpy.array([])
        self._noise_generator = numpy.random.default_rng()

    def step(self) -> None:
        clutter_count = int(
            self._noise_generator.uniform(self._minimum_clutter_count, self._maximum_clutter_count)
        )
        self._features = numpy.insert(
            numpy.array(
                [
                    self._noise_generator.multivariate_normal(
                        self._features_mean, self._features_covariance
                    )
                    for _ in range(clutter_count)
                ]
            ).transpose(),
            0,
            self._identifier,
            0,
        )

    @property
    def features(self) -> numpy.ndarray:
        return self._features


class Simulator:
    def __init__(self, components: typing.List[Component], steps: int) -> None:
        self._components = components
        self._steps = steps

    def run_simulation(self) -> numpy.ndarray:
        features: typing.List[numpy.ndarray] = []
        for _ in range(self._steps):
            for c in self._components:
                c.step()
                features.append(c.features)
        dataset = numpy.concatenate(features, 1)
        # Ensure size is positive non-zero and intensity is between 0 and 1
        return dataset[:, numpy.all([dataset[1] > 0, dataset[2] > 0, dataset[2] < 1], axis=0)]

    class Component(typing.Protocol):
        def step(self) -> None: ...

        @property
        def features(self) -> numpy.ndarray: ...


def plot_cluster(identifier: int, color: typing.List[int]) -> None:
    matplotlib.pyplot.scatter(
        numpy.extract(dataset[0, :] == identifier, dataset[1, :]),
        numpy.extract(dataset[0, :] == identifier, dataset[2, :]),
        color=color,
        alpha=0.1,
    )


def plot_confidence_ellipse(
    mean: numpy.ndarray,
    covariance: numpy.ndarray,
    axes: matplotlib.axes.Axes,
    color: typing.List[int],
    label: str,
) -> None:
    # Ripped straight off the interwebz:
    # https://stackoverflow.com/questions/20126061/creating-a-confidence-ellipse-in-a-scatterplot-using-matplotlib
    def eigsorted(cov):
        vals, vecs = numpy.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    vals, vecs = eigsorted(covariance)
    theta = numpy.degrees(numpy.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * numpy.sqrt(vals)
    ell = matplotlib.patches.Ellipse(
        xy=(mean[0], mean[1]), width=w, height=h, angle=theta, color=color, label=label
    )
    ell.set_facecolor("none")
    axes.add_artist(ell)


if __name__ == "__main__":
    # Define some random clutter spawned in every scan with uniformly random count and normally
    # distributed about some mean feature values and covariance
    CLUTTER_FEATURES_MEAN = numpy.array([5.0, 0.5])
    CLUTTER_FEATURES_COVARIANCE = numpy.array([[4.0, 0.0], [0.0, 0.2]])
    MINIMUM_CLUTTER_COUNT = 3
    MAXIMUM_CLUTTER_COUNT = 10
    CLUTTER_IDENTIFIER = 0
    clutter = Clutter(
        CLUTTER_FEATURES_MEAN,
        CLUTTER_FEATURES_COVARIANCE,
        MINIMUM_CLUTTER_COUNT,
        MAXIMUM_CLUTTER_COUNT,
        CLUTTER_IDENTIFIER,
    )

    TARGET_TIME_DELTA = datetime.timedelta(seconds=1.0)

    # Define a relatively arbitrary target which spawns a single detection every scan with features
    # normally distributed about some mean and covariance
    TARGET_A_ACCELERATION_STANDARD_DEVIATION = 0.1
    TARGET_A_FEATURES_MEAN = numpy.array([10.0, 0.7])
    TARGET_A_FEATURES_COVARIANCE = numpy.array([[5.0, 0.5], [0.5, 0.1]])
    TARGET_A_IDENTIFIER = 1
    target_a_feature_extractor = TargetInvariantTargetFeatureExtractor(
        TARGET_A_FEATURES_MEAN, TARGET_A_FEATURES_COVARIANCE, TARGET_A_IDENTIFIER
    )
    target_a = Target(
        target_a_feature_extractor.extract_features,
        TARGET_TIME_DELTA,
        TARGET_A_ACCELERATION_STANDARD_DEVIATION,
    )

    # Define another relatively arbitrary target which spawns a single detection every scan with
    # features normally distributed about some mean and covariance
    TARGET_B_ACCELERATION_STANDARD_DEVIATION = 0.1
    TARGET_B_FEATURES_MEAN = numpy.array([15.0, 0.4])
    TARGET_B_FEATURES_COVARIANCE = numpy.array([[7.0, 0.3], [0.3, 0.1]])
    TARGET_B_IDENTIFIER = 2
    target_b_feature_extractor = TargetInvariantTargetFeatureExtractor(
        TARGET_B_FEATURES_MEAN, TARGET_B_FEATURES_COVARIANCE, TARGET_B_IDENTIFIER
    )
    target_b = Target(
        target_b_feature_extractor.extract_features,
        TARGET_TIME_DELTA,
        TARGET_B_ACCELERATION_STANDARD_DEVIATION,
    )

    # Generate the dataset
    NUMBER_OF_SCANS = 2000
    simulator = Simulator([clutter, target_a, target_b], NUMBER_OF_SCANS)
    dataset = simulator.run_simulation()

    # Visualize the dataset
    matplotlib.pyplot.subplot(2, 1, 1)
    plot_cluster(0, [1, 0, 0])
    plot_cluster(1, [0, 1, 0])
    plot_cluster(2, [0, 0, 1])
    matplotlib.pyplot.grid()
    matplotlib.pyplot.title("Dataset Colored By Cluster")
    matplotlib.pyplot.xlabel("Size")
    matplotlib.pyplot.ylabel("Intensity")
    matplotlib.pyplot.subplot(2, 1, 2)
    plot_cluster(0, [1, 0, 1])
    plot_cluster(1, [1, 0, 1])
    plot_cluster(2, [1, 0, 1])
    matplotlib.pyplot.grid()
    matplotlib.pyplot.title("Dataset Without Labeled Clusters")
    matplotlib.pyplot.xlabel("Size")
    matplotlib.pyplot.ylabel("Intensity")
    matplotlib.pyplot.savefig("figures/dataset.jpg")

    # Perform expectation maximization to estimate Gaussian mixture (this one gets plotted)
    N_COMPONENTS = 2
    distribution = sklearn.mixture.GaussianMixture(n_components=N_COMPONENTS).fit(
        dataset[1:, :].transpose()
    )
    print(f"N_COMPONENTS={N_COMPONENTS}")
    print(f"Means: {distribution.means_}")
    print(f"Covariances: {distribution.covariances_}")
    print(f"Weights: {distribution.weights_}")

    # Plot results
    COLORS = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 0, 0],
    ]
    matplotlib.pyplot.figure()
    axes = matplotlib.pyplot.subplot(1, 1, 1)
    plot_cluster(0, [0, 0, 1])
    plot_cluster(1, [0, 0, 1])
    plot_cluster(2, [0, 0, 1])
    for i in range(distribution.means_.shape[0]):  # type: ignore
        plot_confidence_ellipse(
            distribution.means_[i, :],  # type: ignore
            distribution.covariances_[i, :, :],  # type: ignore
            axes,
            COLORS[i],
            f"{distribution.weights_[i]}",  # type: ignore
        )
    matplotlib.pyplot.grid()
    matplotlib.pyplot.title("Distributions Overlaid Over Dataset")
    matplotlib.pyplot.xlabel("Size")
    matplotlib.pyplot.ylabel("Intensity")
    matplotlib.pyplot.legend()
    matplotlib.pyplot.savefig("figures/results.jpg")

    # Check BIC and AIC for some component choices
    MAX_N_COMPONENTS = 10
    aic_scores: typing.List[float] = []
    bic_scores: typing.List[float] = []
    x_axis_ticks = range(1, MAX_N_COMPONENTS + 1)
    for i in x_axis_ticks:
        dataset_sans_labels = dataset[1:, :].transpose()
        gm = sklearn.mixture.GaussianMixture(n_components=i).fit(dataset_sans_labels)
        aic_scores.append(gm.aic(dataset_sans_labels))
        bic_scores.append(gm.bic(dataset_sans_labels))

    # Plot values
    matplotlib.pyplot.figure()
    matplotlib.pyplot.subplot(2, 1, 1)
    matplotlib.pyplot.plot(x_axis_ticks, aic_scores)
    matplotlib.pyplot.xticks(x_axis_ticks)
    matplotlib.pyplot.grid()
    matplotlib.pyplot.title("AIC Scores (Lower is Better) vs. Number of Components")
    matplotlib.pyplot.xlabel("Number of Components")
    matplotlib.pyplot.ylabel("AIC Scores")
    matplotlib.pyplot.subplot(2, 1, 2)
    matplotlib.pyplot.plot(x_axis_ticks, bic_scores)
    matplotlib.pyplot.xticks(x_axis_ticks)
    matplotlib.pyplot.grid()
    matplotlib.pyplot.title("BIC Scores (Lower is Better) vs. Number of Components")
    matplotlib.pyplot.xlabel("Number of Components")
    matplotlib.pyplot.ylabel("BIC Scores")
    matplotlib.pyplot.savefig("figures/scores.jpg")
