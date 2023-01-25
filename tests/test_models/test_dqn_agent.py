import os.path
from unittest import TestCase

import numpy as np
import torch
from torch import Tensor

from akademy.models.agents import DQNAgent
from akademy.models.base_models import NetworkBase
from tests.helpers.model_utils import create_mock_trade_environment,\
    generate_random_experience_samples

HERE = os.path.abspath(os.path.dirname(__file__))


class TestDQNAgent(TestCase):
    """
    Test suite to ensure functionality of DQNAgent model meets requirements.
    """
    def setUp(self) -> None:
        """
        Defines values for an agent.
        """
        self.epsilon_min = .001
        self.gamma = .95
        self.batch_size = 64
        self.learning_rate = .001
        self.hidden_n = 512
        self.checkpoint_save_dir = '/'
        self.cpu_mode = True
        self.env = create_mock_trade_environment()
        
        self.agent = DQNAgent(
            action_count=self.env.action_space.n,
            state_shape=self.env.observation_space.shape,
            epsilon_min=self.epsilon_min,
            gamma=self.gamma,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            hidden_n=self.hidden_n,
            checkpoint_save_dir=self.checkpoint_save_dir,
            cpu_mode=self.cpu_mode
        )

        # info to test load/save of models
        self.model_output_name = "test.pt"
        self.model_output_path = os.path.abspath(os.path.join(
            HERE,
            self.model_output_name
        ))

        # add replay experiences
        self._add_agent_replay_experiences()

    def tearDown(self) -> None:
        """
        Series of events executed after these unit tests complete
        """
        try:
            os.remove(self.model_output_path)
        except FileNotFoundError as e:
            # sample file not created
            pass

    def _add_agent_replay_experiences(self, count: int = 64):
        """
        Add some fake experiences to the replay buffer
        """
        # generate fake experiences for the agents replay memory
        self.agent.get_replay_memory().states, \
        self.agent.get_replay_memory().actions, \
        self.agent.get_replay_memory().rewards, \
        self.agent.get_replay_memory().next_states, \
        self.agent.get_replay_memory().dones = \
            generate_random_experience_samples(
                state_shape=self.env.observation_space.shape,
                count=count
            )
        self.agent.get_replay_memory().mem_count = count  # fake the count

    def test_agent_instantiation(self):
        """
        Tests that a new instance of an DQNAgent class object can be
        created and field values are as expected.
        """
        # test instance created
        self.assertTrue(self.agent is not None)
        self.assertTrue(isinstance(self.agent, DQNAgent))

        # test data is as expected
        self.assertEqual(self.agent.action_count, self.env.action_space.n)
        self.assertEqual(self.agent.state_shape, self.env.observation_space.shape)
        self.assertEqual(self.agent.epsilon_min, self.epsilon_min)
        self.assertEqual(self.agent.gamma, self.gamma)
        self.assertEqual(self.agent.batch_size, self.batch_size)
        self.assertEqual(self.agent.learning_rate, self.learning_rate)
        self.assertEqual(self.agent.hidden_n, self.hidden_n)
        self.assertEqual(self.agent.checkpoint_save_dir, self.checkpoint_save_dir)
        self.assertEqual(self.agent.cpu_mode, self.cpu_mode)

    def test_required_fields_missing(self):
        """
        Test that a DQNAgent instance can't be instantiated without all
        required fields (state size, action size)
        """
        # missing observation count
        with self.assertRaises(TypeError):
            a = DQNAgent(action_count=5)

        # missing action count
        with self.assertRaises(TypeError):
            a = DQNAgent(state_shape=(251,))

    def test_has_policy_network(self):
        """
        Test the DQNAgent has a policy network instantiated as expected.
        """
        self.assertTrue(
            self.agent.policy_network is not None
        )

        # doesn't need to be assured as being anymore specific than the
        # base class for any network system.
        self.assertTrue(
            NetworkBase in self.agent.policy_network.__class__.__bases__
        )

        # checks the dimensions of the network as matching of the params
        self.assertEqual(
            self.agent.policy_network.hidden_n,
            self.hidden_n
        )
        self.assertEqual(
            self.agent.policy_network.input_n,
            self.env.observation_space.shape[0]
        )
        self.assertEqual(
            self.agent.policy_network.output_n,
            self.env.action_space.n
        )
        self.assertEqual(
            self.agent.policy_network.learning_rate,
            self.learning_rate
        )
        self.assertEqual(
            self.agent.policy_network.device,
            "cpu" if self.agent.cpu_mode else "cuda"
        )
        
        # ensure components of network exist
        self.assertTrue(self.agent.policy_network.optimizer is not None)
        self.assertTrue(self.agent.policy_network.loss is not None)

    def test_save(self):
        """
        Tests that a valid *.pt PyTorch binary file is created via the save()
        method.
        """
        self.assertTrue(
            not os.path.exists(self.model_output_path)
        )
        self.agent.save(path=self.model_output_path, delete_old=False)
        self.assertTrue(
            os.path.exists(self.model_output_path)
        )

    def test_load(self):
        """
        Tests that a valid *.pt model can be loaded by the agent
        """
        # save an initial model checkpoint
        self.agent.save(path=self.model_output_path, delete_old=False)

        # function returns true on successful execution
        # could be more robust
        self.assertEqual(
            self.agent.load(path=self.model_output_path),
            True
        )

    def test_sample_exp(self):
        """
        Test the agent produces samples from the ExperienceReplay buffer.
        """
        # unpack everything
        states, actions, rewards, next_states, dones = self.agent.sample_exp()

        # assert correct types
        self.assertEqual(type(states), Tensor)
        self.assertEqual(type(actions), Tensor)
        self.assertEqual(type(rewards), Tensor)
        self.assertEqual(type(next_states), Tensor)
        self.assertEqual(type(dones), Tensor)

        # assert correct counts
        self.assertEqual(len(states), self.agent.batch_size)
        self.assertEqual(len(actions), self.agent.batch_size)
        self.assertEqual(len(rewards), self.agent.batch_size)
        self.assertEqual(len(next_states), self.agent.batch_size)
        self.assertEqual(len(dones), self.agent.batch_size)

        # assert correct data types
        # revisit later to ensure not limiting in flexibility.p
        self.assertEqual(states[0].dtype, torch.float32)
        self.assertEqual(actions[0].dtype, torch.int64)
        self.assertEqual(rewards[0].dtype, torch.float32)
        self.assertEqual(next_states[0].dtype, torch.float32)
        self.assertEqual(dones[0].dtype, torch.bool)

    def test_infer(self):
        """
        Tests that the inference method of the model works via the following
        assertions:
            1. produces the expected output
            2. model weights are unchanged
        """
        # generate an observation of random values in the dimesion of:
        # (1, env.window * 5 + 1)
        obv = np.random.random_sample((1, self.env.observation_space.shape[0]))

        # snapshot of the network weights before inference
        pre_state_weights = self.agent.policy_network.state_dict().__str__()

        # generated an inference, check is expected value range, type
        action = self.agent.infer(state=obv)
        self.assertTrue(action is not None)
        self.assertTrue(type(action) == int)
        self.assertTrue(action in list(range(self.env.action_space.n)))

        # confirms that inference generation doesn't affect model weights
        self.assertEqual(
            pre_state_weights,
            self.agent.policy_network.state_dict().__str__()
        )

        # generate multiple actions to assert all fall within expected range
        for _ in range(64):
            obv = np.random.random_sample(
                (1, self.env.observation_space.shape[0])
            )
            action = self.agent.infer(obv)
            self.assertTrue(
                action in list(range(self.env.action_space.n))
            )

    def test_train(self):
        """
        Test the Agents' <train> method completes via the following observations
        of state change in the Agent + policy network:
            1. epsilon is decremented
            2. loss is returned
            3. network weights change *
        * note: this might not be great for larger models. Not an issue here.
        References:
            Checking PyTorch state Dict's
            https://discuss.pytorch.org/t/check-if-models-have-same-weights/4351/10
        """
        # should be the same here
        state_dict = self.agent.policy_network.state_dict().__str__()
        state_dict_2 = self.agent.policy_network.state_dict().__str__()
        self.assertEqual(
            state_dict,
            state_dict_2
        )

        # not train and assert model weights are updated
        loss = self.agent.train()
        self.assertTrue(loss is not None)
        state_dict_3 = self.agent.policy_network.state_dict().__str__()
        self.assertNotEqual(
            state_dict,
            state_dict_3
        )
