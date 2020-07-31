r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=256, seq_len=64,
        h_dim=256, n_layers=3, dropout=0.5,
        learn_rate=0.001, lr_sched_factor=0.5, lr_sched_patience=2,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    # learn_rate = 0.1
    # seq_len = 45
    # batch_size = 15
    # lr_sched_factor = 0.1
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = .0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "A literary masterpiece is a work of literature that is considered to be outstanding in terms of its artistry and technique, and is held in high esteem as an original work to be read and studied. A literary masterpiece can take the form of any written work, including a poem, short story, play or novel."
    temperature = 0.7
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**
Training over the entire corpus could cause the net-params to fixed over a bad local optimum while using
sequences could improve the generalization of the params.
In addition to that, back prop over the whole text will result in anishing gradients, making the network hard to train, therefore
we splited the course into small sequences.

"""

part1_q2 = r"""
**Your answer:**
The hidden state is saved between batches on the same epoch. Therefore, the network can remember att tthe data back
to the epoch beginning while training. In addition to that,we update our model parameters in each batch.
Those parameters are not reset during the entirety of the training.
"""

part1_q3 = r"""
**Your answer:**
We want to predict the next char given the sequence that based on the previous sequences, therefore,
we would like to keep the original order between the sequences.
The order is very important in this case.
"""

part1_q4 = r"""
**Your answer:**

1. We reduced the temperature when sampling to reduce the variety of generated text.
Performing softmax on lower temperature values makes the RNN more confident - it needs less input to
activate the output layer, but also more conservative in its samples.

2. The temperature increases the sensitivity to low probability candidates.
Using a high temperature produces a lower probability distribution over the classes,
resulting in more diversity.

3. When the temperature is low we reduce the random effect of selecting the next char.
As a result, the generated text is more "systematic" with less variety.
"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0,
        h_dim=0, z_dim=0, x_sigma2=0,
        learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 2
    hypers['h_dim'] = 128
    hypers['z_dim'] = 64
    hypers['x_sigma2'] = 0.001
    hypers['learn_rate'] = 0.001
    hypers['betas'] = (0.85, 0.999)
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**
The x_sigma2 is having its effect over the loss function. The bigger x_sigma2 is the lower the
data-reconstruction loss effect over the loss (the KL-divergence loss has more influance) and vise-versa.
While the x_sigma2 is low - the decoder will learn to reconstruct the data and won't construct
the latent space. While the x_sigma2 is heigh - the data-reconstruction loss has minor effect
and will learn to reconstruct the data badly.

"""

part2_q2 = r"""
**Your answer:**
1. The reconstruction loss purpose is to learn the decoder to reconstruct
the images properly. The KL divergence loss purpose is to minimize the difference between the p and
the q distributuions.
2. The reparametrization trick is used to calculate z (latent representation). Therefore,
when we use KL loss we bias the latet space in order to get close to the z distributuion.
3. That will make the latent representation be more "personal" to the data. Meaning that
different data will have different latent representation.

"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0, z_dim=0,
        data_label=0, label_noise=0.0,
        discriminator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 2
    hypers['z_dim'] = 256
    hypers['data_label'] = 0
    hypers['label_noise'] = 0.3
    hypers['discriminator_optimizer']['type'] = 'SGD'
    hypers['discriminator_optimizer']['lr'] = 0.001
    hypers['generator_optimizer']['type'] = 'Adam'
    hypers['generator_optimizer']['lr'] = 0.0001
    hypers['generator_optimizer']['betas'] = (0.5, 0.99)
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**

When producing image from the generator (sampling) while its trained then we need to back prop
accourding to the loos calculated on the specific sampls that were generated (from the forward step).
But, when training the discriminator and we want to calculate only the back prop according to the
discriminator without using the gradients of the generator.

"""

part3_q2 = r"""
**Your answer:**
1. When training the models, as we saw, the generator and the discriminator losses won't
converge and keep oscillating. That means that the point of stopping the training is
when the looses won't optemize based on each other.

2. That means that the generator learned to imitate the real images almost perfectly
or at least in such a good way that the discriminator can't distinguish whether the image
is real or not.
"""

part3_q3 = r"""
The main differences between the two results are:
The VAE output is quite blurry and not so realistic, in contrast a  well-trained GAN output should be much more realistic and sharp.
We think this is because VAE learns a complex distribution of the data, and it tries to fit it via a Gaussian distribution.
A trained GAN generates images from the same distribution of the real world data, therefore the pictures are much more
realistic.

"""

# ==============
