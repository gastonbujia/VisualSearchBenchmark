# Metrics
Three different metrics are computed, with a focus on efficiency and scanpath similarity against humans:
* **Cumulative performance** (*AUCperf*): Proportion of targets found (vertical axis) for a given number of fixations (horizontal axis). The Area Under the Curve is computed. To measure the similarity with humans, the final score is computed by: *1 - |AUCperf(subjects) - AUCperf(model)|*
* **MultiMatch** (*AvgMM*): Compares two given scanpaths in several dimensions, by treating them as geometrical vectors in a two-dimensional space. Models' scanpaths are compared against those of human subjects and the average is computed across all dimensions, with the exception of time. See https://multimatch.readthedocs.io for more information on the algorithm.
* **Human Scanpath Prediction** (*AUChsp* and *NSShsp*): Given the scanpath of a human subject, each model attempts to predict where the next fixation is going to land. This is done for each fixation in the scanpath (with the exception of the first one) and allows for the computation of the Area Under the Curve and Normalized Saliency Scanpath (NSS). Results are averaged across all fixations. This metric originated from the paper [State-of-the-art in Human Scanpath Prediction](https://arxiv.org/abs/2102.12239) and we believe this to be its first implementation in the problem of visual search.