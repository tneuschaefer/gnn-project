# **Rationale for Using Conditional Wasserstein GAN in EEG Augmentation**

## **Introduction**
In our project, we address two major challenges in EEG-based brain-computer interface (BCI) research: the scarcity of labeled EEG data and the need for reliable data augmentation.
We have chosen to use a Conditional Wasserstein GAN with Gradient Penalty (WGAN-GP) to generate synthetic EEG epochs conditioned on emotion labels (e.g., “positive” vs. “negative”).
This approach is inspired by both theoretical insights and empirical evidence from recent literature.
## **Why Conditional Wasserstein GAN?**
Standard GANs, introduced by Goodfellow et al., have been successful in generating realistic images but suffer from training instability and mode collapse.
These issues are exacerbated in the high-dimensional, noisy domain of EEG signals. In contrast, the Wasserstein GAN framework replaces the Jensen-Shannon divergence with the Wasserstein distance—a metric that provides more meaningful gradients during training. The incorporation of a gradient penalty (WGAN-GP)
enforces the Lipschitz constraint without resorting to weight clipping, further stabilizing training and improving sample quality.

Additionally, the _conditional_ variant allows us to incorporate label information directly into the generation process. This is critical for our application because:

- **Controlled Generation:** By conditioning on emotion labels, our GAN can generate class-specific EEG patterns. For instance, by concatenating one-hot encoded labels with the latent noise vector, the generator learns to produce synthetic EEG epochs that are distinctly “positive” or “negative.”
- **Improved Augmentation:** This control over the generated output ensures that the synthetic data better match the statistical distribution of the real, class-specific EEG data. In turn, this targeted augmentation helps improve downstream classification performance.

## **Differences from Standard GANs**
- **Stability and Quality:** Standard GANs often face issues like mode collapse, where the generator produces limited varieties of outputs. The Wasserstein loss in WGAN-GP provides smoother, more stable gradients, resulting in higher-quality synthetic EEG signals.
- **Gradient Penalty:** Unlike standard GANs that may use weight clipping (which can hinder model capacity), WGAN-GP uses a gradient penalty that ensures the discriminator (critic) satisfies the Lipschitz constraint more effectively.
- **Conditional Input:** While a standard GAN generates data without control, our conditional GAN explicitly integrates label information in both the generator and discriminator. This results in a model that can generate specific classes of EEG signals—a feature crucial for data augmentation in emotion recognition tasks.

## **Our Conditioning Strategy**
We condition our GAN on binary emotion labels (e.g., “positive” vs. “negative”) as follows:
- **Generator:** The input consists of a 100-dimensional noise vector concatenated with a one-hot encoded label (2-dimensional). This combined vector is mapped via a fully-connected layer and further upsampled through transposed convolutions to produce an EEG epoch of shape (63 channels, 1001 timepoints).
- **Discriminator:** The real or synthetic EEG data is concatenated with the replicated one-hot label along the channel dimension. This ensures the discriminator learns to assess not only the authenticity of the EEG signal but also its consistency with the provided class label.

# **Supporting Literature**
Our approach is well-founded in recent research:

- The paper “[Deep Convolutional Neural Network-Based Visual Stimuli Classification Using Electroencephalography Signals of Healthy and Alzheimer’s Disease Subjects](https://pmc.ncbi.nlm.nih.gov/articles/PMC8950142/)” demonstrates the potential of deep convolutional architectures in EEG analysis, providing insight into feature extraction from EEG signals.
- “[Generative adversarial networks in EEG analysis: an overview](https://jneuroengrehab.biomedcentral.com/articles/10.1186/s12984-023-01169-w)” reviews the application of GANs in EEG analysis, highlighting the benefits of conditional models and the improvements in training stability when using Wasserstein loss.
- “[Data Augmentation for EEG-Based Emotion Recognition Using Generative Adversarial Networks](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2021.723843/full)” provides empirical evidence that conditional GANs can significantly improve emotion recognition performance through data augmentation.