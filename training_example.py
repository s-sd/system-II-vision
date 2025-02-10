import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from unet import get_unet, get_convnet
from alexnet import get_alexnet
from tqdm import tqdm

# =============================================================================
# Data preperation
# =============================================================================

def add_gaussian_noise(images, mean=0, std=0.4, clip_range=[0, 1]):
    noise = np.random.normal(mean, std, size=images.shape)
    noised_images = images + noise
    noised_images = np.clip(noised_images, clip_range[0], clip_range[1])
    return noised_images
    

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# norm to [-1, +1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# pad to shape (32, 32)
pad_width = ((0, 0), (2, 2), (2, 2))  # Pad 2 elements before and after along both dimensions
train_images = np.pad(train_images, pad_width, mode='constant', constant_values=0)
test_images = np.pad(test_images, pad_width, mode='constant', constant_values=0)

# add noise to images, these noised versions act as images
noised_train_images = add_gaussian_noise(train_images)
noised_test_images = add_gaussian_noise(test_images)

# threshold the images to act as segmentation masks for the noised images
train_images[train_images>=0.5] = +1
train_images[train_images<0.5] = 0
test_images[test_images>=0.5] = +1
test_images[test_images<0.5] = 0

num_val = 4_000

# training data
x_train = deepcopy(noised_train_images[:num_val])
y_train = deepcopy(train_images[:num_val])
labels_train = deepcopy(train_labels[:num_val])

x_train = np.expand_dims(x_train, axis=-1).astype(np.float32)
y_train = np.expand_dims(y_train, axis=-1).astype(np.float32)

# val data
x_val = deepcopy(noised_train_images[num_val:])
y_val = deepcopy(train_images[num_val:])
labels_val = deepcopy(train_labels[num_val:])

x_val = np.expand_dims(x_val, axis=-1).astype(np.float32)
y_val = np.expand_dims(y_val, axis=-1).astype(np.float32)

# testing data
x_test = deepcopy(noised_test_images)
y_test = deepcopy(test_images)
labels_test = deepcopy(test_labels)

x_test = np.expand_dims(x_test, axis=-1).astype(np.float32)
y_test = np.expand_dims(y_test, axis=-1).astype(np.float32)
# y in range (0, 1)
# x in range (0, 1)

# =============================================================================
# Splitting into separate digits
# =============================================================================

def get_digit_dataset(x, y, labels):
    digit_datasets = []
    for digit in range(10):
        digit_indices = np.where(labels == digit)[0]
        digit_x = x[digit_indices]
        digit_y = y[digit_indices]
        digit_datasets.append((digit_x, digit_y))
    return digit_datasets

train_dataset = get_digit_dataset(x_train, y_train, labels_train)
val_dataset = get_digit_dataset(x_val, y_val, labels_val)
test_dataset = get_digit_dataset(x_test, y_test, labels_test)


# =============================================================================
# Adversarial training for the System I module
# =============================================================================


def get_data_batch(dataset, batch_size, digit):
    x, y = dataset[digit]
    shuffled_indexes = np.random.permutation(len(x))
    batch_indexes = shuffled_indexes[:batch_size]
    x_batch = x[batch_indexes]
    y_batch = y[batch_indexes]
    return x_batch, y_batch

# train_data_gen = data_generator(x_train, y_train, batch_size, digits=[0, 1, 2, 3, 4, 5])
# val_data_gen = data_generator(x_val, y_val, batch_size, digits=[6, 7])

generator = get_convnet(x_train.shape[1:], 1)
discriminator = get_alexnet(x_train.shape[1:], x_test.shape[1:])

def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = tf.keras.layers.Input(x_train.shape[1:])
    generated_y = generator(gan_input)
    gan_output = discriminator([gan_input, generated_y])
    gan = tf.keras.models.Model(gan_input, gan_output)
    return gan


gan = build_gan(generator, discriminator)

discriminator_trainable = get_alexnet(x_train.shape[1:], x_test.shape[1:])

# always set discriminator trainable to true when training only discriminator 
# and back to false when training gen using gan

discriminator_trainable.compile(loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            metrics=['accuracy'])
discriminator.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                      optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5))
gan.compile(loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5))

batch_size = 32

epochs = 5
num_steps = 4
val_steps = 8
num_iterations = 96
epsilon = 0.02

def mse(y_1, y_2):
    return (y_1 - y_2)**2

train_digits = [0, 1, 2, 3, 4, 5]
val_digits = [6, 7]
test_digits = [8, 9]

# for demo, can stop training at around 0.01 validation discrimination; approx 100 iter
for i in range(num_iterations):
    
    print(f'\nIteration {i+1} / {num_iterations}')
    
    generator_params = generator.get_weights()
    discriminator_params = discriminator.get_weights()
    discriminator_trainable.set_weights(discriminator_params)
    
    digit = int(np.random.choice(train_digits))

    for _ in tqdm(range(num_steps)): # k steps on one task
        
        x_train_batch, y_train_batch = get_data_batch(dataset=train_dataset, batch_size=batch_size, digit=digit) 
        
        # Train discriminator
        fake_y = generator.predict(x_train_batch)
        real_y = y_train_batch
        combined_y = np.concatenate([real_y, fake_y], axis=0)
        combined_x = np.concatenate([x_train_batch, x_train_batch], axis=0)
        
        labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
        labels += 0.05 * np.random.random(labels.shape)  # Add noise to labels
        
        discriminator_loss = discriminator_trainable.train_on_batch([combined_x, combined_y], labels)
        
        discriminator.set_weights(discriminator_trainable.get_weights())

        # Train generator
        misleading_labels = np.ones((batch_size, 1))
        generator_loss = gan.train_on_batch(x_train_batch, misleading_labels)

    # discriminator_params + eps * (discriminator.get_weights() - discriminator_params)
    discriminator_params = [param + epsilon*(task_param - param) for param, task_param in zip(discriminator_params, discriminator.get_weights())]
    discriminator.set_weights(discriminator_params)
    
    # generator_params + eps * (generator.get_weights() - generator_params)
    generator_params = [param + epsilon*(task_param - param) for param, task_param in zip(generator_params, generator.get_weights())]
    generator.set_weights(generator_params)
    
    #########################
    
    epoch_similarity_scores = []
    epoch_discrimination = []
    for _ in range(val_steps):
        x_val_batch, y_val_batch = get_data_batch(val_dataset, batch_size=batch_size, digit=np.random.choice(val_digits))
        y_val_batch_predicted = generator.predict(x_val_batch)
        
        similarity_score = mse(np.array(y_val_batch), np.array(y_val_batch_predicted))
        epoch_similarity_scores.append(similarity_score)
        
        real_discriminator = discriminator.predict([x_val_batch, y_val_batch])
        fake_discriminator = discriminator.predict([x_val_batch, y_val_batch_predicted])
        discrimination = np.array(real_discriminator) - np.array(fake_discriminator) # should be close to 1 as improves
        epoch_discrimination.append(discrimination)
        
    mean_similarity_scores = np.mean(epoch_similarity_scores)
    mean_discrimination = np.mean(epoch_discrimination)
    
    print(f'Validation Similarity: {mean_similarity_scores}, Validation Discrimination: {mean_discrimination}')
    
# index = 8
# plt.imshow(x_val_batch[index])
# plt.imshow(y_val_batch[index])
# plt.imshow(y_val_batch_predicted[index])

# plt.imshow(y_val_batch_predicted[index]>0.501)

# maybe stopped early at 0.9 discrimination
# approx 6 epochs
# but i keeps improvign afterwards, the losses and metrics change slowly after an initial fall
# for demo we can stop early and demo between a few digits
# set weights can be used for reptile
# gather from all different digits and then set weights as weighted sum


# =============================================================================
# Adapting the System I module using few samples - 4 samples from digit 8
# =============================================================================

digit = 8
num_samples = 4

x_test_batch, y_test_batch = get_data_batch(test_dataset, batch_size=num_samples, digit=digit)

# adapt the two networks using 8 samples

num_iterations_adaptation = 16 # increase if required

for _ in range(num_iterations_adaptation):
    
    fake_y = generator.predict(x_test_batch)
    real_y = y_test_batch
    combined_y = np.concatenate([real_y, fake_y], axis=0)
    combined_x = np.concatenate([x_test_batch, x_test_batch], axis=0)
    
    labels = np.concatenate([np.ones((num_samples, 1)), np.zeros((num_samples, 1))])
    labels += 0.05 * np.random.random(labels.shape)  # Add noise to labels
    
    discriminator_loss = discriminator_trainable.train_on_batch([combined_x, combined_y], labels)
    
    discriminator.set_weights(discriminator_trainable.get_weights())

    # Train generator
    misleading_labels = np.ones((num_samples, 1))
    generator_loss = gan.train_on_batch(x_test_batch, misleading_labels)


# predicted_segmentation = generator.predict(x_test_batch[0:1])
# plt.imshow(predicted_segmentation[0])
# plt.imshow(predicted_segmentation[0]>0.61) # select a threshold, not required if training every step to convergence, just required for demo

# =============================================================================
# System II thinking using self-play RL
# =============================================================================
# assume a single unlabelled sample needs to be labelled by system II

import gym

class Type2Thinking(gym.Env):
    
    def __init__(self, sample, generator, discriminator):
        
        self.competitor = None
        
        self.sample = np.squeeze(sample)
        self.generator = generator
        self.discriminator = discriminator
        
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(*np.shape(self.sample), 2))
        self.action_space = gym.spaces.Box(low=0, high=1, shape=np.shape(self.sample))
                
        self.predicted_segmentation = None
        self.threshold = 0.508 # this might need to be adjusted for the demo, if using few training steps above, however, for full training of gan, we can leave it at 0.5
        
        self.proportion_flips_allowed = 0.01
        
        self.termination_steps = 1024 # approx proportion_flips * prod(action_space) * 100
        
        self.competitor_observations = []
        self.competitor_actions = []
        self.competitor_rewards = []
        self.competitor_dones = []
        
        self.step_counter = 0
    
    def _get_observation(self, sample, predicted_segmentation):
        observation = np.concatenate([np.expand_dims(sample, axis=-1), np.expand_dims(predicted_segmentation, axis=-1)], axis=-1)
        return observation
    
    def _predict_generator(self, sample):
        predicted_map = np.squeeze(self.generator.predict(np.expand_dims(np.expand_dims(sample, axis=-1), axis=0)))
        predicted_segmentation = predicted_map > self.threshold
        return predicted_segmentation
    
    def _predict_discriminator(self, sample, predicted_segmentation):
        return np.squeeze(self.discriminator.predict([np.expand_dims(np.expand_dims(sample, axis=-1), axis=0),
                                                      np.expand_dims(np.expand_dims(predicted_segmentation, axis=-1), axis=0)]))
    
    def _logical_operation(self, action, segmentation):
        in1 = np.array(action, dtype=bool)
        in2 = np.array(segmentation, dtype=bool)
        result = (in1 & ~in2) | (~in1 & in2)
        return np.array(result, dtype=np.uint8)
    
    def _apply_action(self, action, predicted_segmentation):
        predicted_segmentation_out = self._logical_operation(action, predicted_segmentation)
        return predicted_segmentation_out
    
    def _round_top_proportion(self, array, proportion):
        percentage = proportion * 100    
        threshold = np.percentile(array, 100 - percentage) # Calculate the threshold value for the top n% of elements
        rounded_arr = np.where(array >= threshold, 1, 0) # Round elements greater than or equal to the threshold to 1, and the rest to 0
        return rounded_arr
    
    def step(self, action):
        
        action = np.reshape(action, env.action_space.shape)
        action = self._round_top_proportion(action, proportion=self.proportion_flips_allowed)
        
        previous_observation = self._get_observation(self.sample, self.predicted_segmentation)
        
        competitor_action = self.competitor.predict(previous_observation)[0]
        competitor_action = np.reshape(competitor_action, env.action_space.shape)
        
        
        competitor_segmentation = self._apply_action(competitor_action, self.predicted_segmentation)
        competitor_discrimination = self._predict_discriminator(self.sample, competitor_segmentation)
        
        agent_segmentation = self._apply_action(action, self.predicted_segmentation)
        agent_discrimination = self._predict_discriminator(self.sample, agent_segmentation)
        
        reward = agent_discrimination - competitor_discrimination
        
        # pick the better segmentation
        # for more complex problems set self.proportion_flips_alowed to 0.01
        if reward < 0:
            self.predicted_segmentation = competitor_segmentation
            self.competitor_rewards.append(reward * -1)        
        else:
            self.predicted_segmentation = agent_segmentation
            
        observation = self._get_observation(self.sample, self.predicted_segmentation)
        
        done = False
        
        self.competitor_observations.append(previous_observation)
        self.competitor_actions.append(competitor_action)
        self.competitor_dones.append(done)
        
        # print(reward)
        
        self.step_counter += 1
        
        if self.step_counter > self.termination_steps:
            done = True
        
        return observation, reward, done, {}
        
    def reset(self):
        self.step_counter = 0
        self.predicted_segmentation = self._predict_generator(self.sample)
        observation = self._get_observation(self.sample, self.predicted_segmentation)
        return observation
    
    def set_competitor(self, new_competitor):
        self.competitor = new_competitor


# for simplicity we will use vanialla self-play here instead of fictitious (agent competes with a copy of itself - the copy is lags behind the agent)
# for simplicity we also only use the agents experiences for training
# for more advanced problems we can use the collected competitor_observations, competitor_actions, competitor_rewards and competitor_dones to train too

digit = 8
num_samples = 1
sample, ground_truth = get_data_batch(test_dataset, batch_size=num_samples, digit=digit)
env = Type2Thinking(sample, generator, discriminator)
# obs_1 = env.reset()

# plt.imshow(obs[:, :, 1])

# start with a  dummy competitor
class dummy_func():
    
    def __init__():
        pass
    
    def predict(inputs):
        return [np.zeros(env.action_space.shape), 0]

env.competitor = dummy_func




observation = env.reset()

obs_2, rew, don, _ = env.step(env.action_space.sample()); print(env.step_counter); print(don)

plt.imshow(obs_2[:, :, 1])

# =============================================================================
# RL self-play training for the System II module
# =============================================================================

# class FlattenedActions(gym.ActionWrapper):
#     def __init__(self, env):
#         super().__init__(env)
#         self.action_space = gym.spaces.MultiBinary(np.prod(env.action_space.shape))
    
#     def action(self, act):
#         return np.reshape(act, newshape=np.prod(act.shape))


from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

def env_creator():
    env = Type2Thinking(sample, generator, discriminator)
    env.competitor = dummy_func
    return env

# wrapped_env = FlattenedActions(env)

vec_env = DummyVecEnv([env_creator])

# use a pre-trained model here to only adapt it rather than train from scratch
model = PPO('MlpPolicy', vec_env, verbose=0, n_steps=env.termination_steps*4, ent_coef=0.01) # for more complex problems we use different architectures


competitor_update_frequency = env.termination_steps * 8
num_iterations = 32 # increase to 1e6 for real run

# should converge to around 0 as competitor also improves
for iteration in range(num_iterations):
    print(f'Self-play iteration: {iteration+1} / {num_iterations}')
    model.learn(competitor_update_frequency)
    vec_env.env_method('set_competitor', model)
    # vec_env.competitor = model
    # model.env = wrapped_env
    rewards = vec_env.get_attr('competitor_rewards')[0]
    rewards = rewards[-1024:]
    print('Reward: ', np.mean(rewards))

final_segmentation = vec_env.get_attr('predicted_segmentation')[0]

plt.imshow(final_segmentation)


# once we have final segmentation, we adapt the gan again using the original adaptation batch plus the obtained segmentation

# in real training, we can have 4-8 samples for the RL tranining
# subsequently, the training for the next 4-8 would be much faster than for the first batch
# we just samples between 4-8 samples on each environment reset
# see other scripts for code



