# Active-Power-Malicious-Attack-Identification
This repository implements an anomaly detection system for wind turbine power data using an autoencoder in Python with Keras. The system focuses on identifying malicious grid manipulations that attempt to disrupt power generation by injecting harmonic signals into the active power data.

The provided readme file covers the details of a project that uses an autoencoder for anomaly detection in wind turbine active power data. Here's a breakdown of the key sections:

**Title and Authors:**

* Title: "Active Power Malicious Attack Identification"
* Authors: G.E. North Piegan, B.Ahamed Sheik Mustafa, and A. Mohamed Hag

**Abstract:**

* Briefly describes the paper's objective: using an autoencoder in Python to identify malicious harmonic injections in wind turbine active power data.
* Mentions data transformation from time domain to frequency domain for better visualization and suitability for the autoencoder.
* Highlights the successful detection of injected harmonics across the spectrum.

**Challenges:**

* The initial method of identifying slow-running wind turbines using reactive power wasn't feasible due to low-frequency content in the data.
* The solution shifted to detecting malicious attacks using active power data in the frequency domain.

**Literature Review:**

* Mentions the autoencoder's origin and its use for tasks like dimension reduction and anomaly detection.
* Cites references for autoencoder applications in anomaly detection for gas turbines, electric drives, and intrusion detection systems.

**Methodology:**

* Explains the advantage of transforming data to the frequency domain for feature detection and easier injection implementation.
* Briefly explains the Fast Fourier Transform (FFT) for converting time-series data to frequency components.
* Provides an overview of the autoencoder and its function of learning efficient data representations.

**Results and Discussions:**

* Describes the implementation using Keras in Python.
* Explains the selection of active power data from a specific wind turbine and its transformation to the frequency domain.
* Discusses the advantage of using the frequency domain for better visualization of injected signals.
* Explains training the autoencoder on the original active power data and using an LSTM model.
* Mentions selecting an optimizer and the number of epochs (training iterations) based on minimizing the reconstruction error. 
* Shows plots comparing the original and reconstructed signals, demonstrating a low error rate.
* Discusses setting a threshold for anomaly detection based on the standard deviation or the maximum error.
* Highlights the autoencoder's ability to pick up anomalies regardless of the magnitude or location of the injected harmonics, as long as they exceed the threshold.

**Conclusion and Future Work:**

* Summarizes the success of the autoencoder in identifying malicious harmonic injections in wind turbine active power data.
* Acknowledges the need for future work in identifying attacks within the typical error range between the original and reconstructed signals.
* Suggests exploring more advanced autoencoders for reducing reconstruction error.

**References:**

* Lists references used in the paper, including the origin of autoencoders, applications in anomaly detection, and resources on the Fast Fourier Transform.

**Appendix I: Python Code**

* Provides the Python code used for the implementation, including data loading, transformation, autoencoder creation and training, anomaly detection, and plotting functionalities.

**Note:**

* Replace "Active Power Malicious Attack Identification" with the actual paper reference details.
* The code utilizes various libraries like TensorFlow, Keras, NumPy, and Pandas for data manipulation, modeling, and visualization.

This communicates the project's purpose, methodology, results, and future directions. It also provides a comprehensive reference list and includes the Python code for users who want to understand the implementation details.
