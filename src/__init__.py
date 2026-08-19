"""Library code for the BayesianExoAdaptation pipeline.

Modules:
    loader             UCI HAR dataset loading and validation.
    hapt_loader        HAPT (postural transitions) loader, used for open-set evaluation.
    subject_split      Subject-wise source/target partitioning and shift-proxy computation.
    bayesian           FeatureClassifier, last-layer Laplace approximation, BALD decomposition.
    har_train          Source MAP training and per-subject evaluation for the HAR MLP.
    im_adapt           Information-maximization source-free adaptation (SHOT-IM / U-SFAN).
    calibration        ECE, NLL, Brier, misclassification AUROC, accuracy-vs-coverage.
    toy                Synthetic 2D three-class track replicating the paper's Fig. 4.
    mnist_check        MNIST vs. Fashion-MNIST / Rotated-MNIST sanity check.
"""
