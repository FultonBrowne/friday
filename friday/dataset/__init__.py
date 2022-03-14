import torch
from torch.utils.data import Dataset
from math import ceil
from torch.utils.data import Dataset
from typing import Dict, Optional, Sequence, Tuple, Union
class BitPatternSet(Dataset):
    """
    Binary multiple instance learning (MIL) data set comprising bit patterns as instances,
    with implanted bit patterns unique to one of the classes.
    """

    def __init__(self, num_bags: int, num_instances: int, num_signals: int, num_signals_per_bag: int = 1,
                 fraction_targets: float = 0.5, num_bits: int = 8, dtype: torch.dtype = torch.float32,
                 seed_signals: int = 43, seed_data: int = 44):
        """
        Create new binary bit pattern data set conforming to the specified properties.
        :param num_bags: amount of bags
        :param num_instances: amount of instances per bag
        :param num_signals: amount of unique signals used to distinguish both classes
        :param num_signals_per_bag: amount of unique signals to be implanted per bag
        :param fraction_targets: fraction of targets
        :param num_bits: amount of bits per instance
        :param dtype: data type of instances
        :param seed_signals: random seed used to generate the signals of the data set (excl. samples)
        :param seed_data: random seed used to generate the samples of the data set (excl. signals)
        """
        super(BitPatternSet, self).__init__()
        assert (type(num_bags) == int) and (num_bags > 0), r'"num_bags" must be a positive integer!'
        assert (type(num_instances) == int) and (num_instances > 0), r'"num_instances" must be a positive integer!'
        assert (type(num_signals) == int) and (num_signals > 0), r'"num_signals" must be a positive integer!'
        assert (type(num_signals_per_bag) == int) and (num_signals_per_bag >= 0) and (
                num_signals_per_bag <= num_instances), r'"num_signals_per_bag" must be a non-negative integer!'
        assert (type(fraction_targets) == float) and (fraction_targets > 0) and (
                fraction_targets < 1), r'"fraction_targets" must be in interval (0; 1)!'
        assert (type(num_bits) == int) and (num_bits > 0), r'"num_bits" must be a positive integer!'
        assert ((2 ** num_bits) - 1) > num_signals, r'"num_signals" must be smaller than "2 ** num_bits - 1"!'
        assert type(seed_signals) == int, r'"seed_signals" must be an integer!'
        assert type(seed_data) == int, r'"seed_data" must be an integer!'

        self.__num_bags = num_bags
        self.__num_instances = num_instances
        self.__num_signals = num_signals
        self.__num_signals_per_bag = num_signals_per_bag
        self.__fraction_targets = fraction_targets
        self.__num_targets = min(self.__num_bags, max(1, ceil(self.__num_bags * self.__fraction_targets)))
        self.__num_bits = num_bits
        self.__dtype = dtype
        self.__seed_signals = seed_signals
        self.__seed_data = seed_data
        self.__data, self.__targets, self.__signals = self._generate_bit_pattern_set()

    def __len__(self) -> int:
        """
        Fetch amount of bags.
        :return: amount of bags
        """
        return self.__num_bags

    def __getitem__(self, item_index: int) -> Dict[str, torch.Tensor]:
        """
        Fetch specific bag.
        :param item_index: specific bag to fetch
        :return: specific bag as dictionary of tensors
        """
        return {r'data': self.__data[item_index].to(dtype=self.__dtype),
                r'target': self.__targets[item_index].to(dtype=self.__dtype)}

    def _generate_bit_pattern_set(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate underlying bit pattern data set.
        :return: tuple containing generated bags, targets and signals
        """
        torch.random.manual_seed(seed=self.__seed_signals)

        # Generate signal patterns.
        generated_signals = torch.randint(low=0, high=8, size=(self.__num_signals, self.__num_bits))
        check_instances = True
        while check_instances:
            generated_signals = torch.unique(input=generated_signals, dim=0)
            generated_signals = generated_signals[generated_signals.sum(axis=1) != 0]
            missing_signals = self.__num_signals - generated_signals.shape[0]
            if missing_signals > 0:
                generated_signals = torch.cat(tensors=(
                    generated_signals, torch.randint(low=0, high=2, size=(missing_signals, self.__num_bits))), dim=0)
            else:
                check_instances = False

        # Generate data and target tensors.
        torch.random.manual_seed(seed=self.__seed_data)
        generated_data = torch.randint(low=0, high=2, size=(self.__num_bags, self.__num_instances, self.__num_bits))
        generated_targets = torch.zeros(size=(self.__num_bags,), dtype=generated_data.dtype)
        generated_targets[:self.__num_targets] = 1

        # Check invalid (all-zero and signal) instances and re-sample them.
        check_instances = True
        while check_instances:
            invalid_instances = (generated_data.sum(axis=2) == 0).logical_or(
                torch.sum(torch.stack([(generated_data == _).all(axis=2) for _ in generated_signals]), axis=0))
            if invalid_instances.sum() > 0:
                generated_data[invalid_instances] = torch.randint(
                    low=0, high=2, size=(invalid_instances.sum(), self.__num_bits))
            else:
                check_instances = False

        # Re-implant signal into respective bags.
        for data_index in range(self.__num_targets):
            implantation_indices = []
            for _ in range(self.__num_signals_per_bag):
                while True:
                    current_implantation_index = torch.randint(low=0, high=generated_data.shape[1], size=(1,))
                    if current_implantation_index not in implantation_indices:
                        implantation_indices.append(current_implantation_index)
                        break
                current_signal_index = torch.randint(low=0, high=generated_signals.shape[0], size=(1,))
                generated_data[data_index, current_implantation_index] = generated_signals[current_signal_index]

        return generated_data, generated_targets, generated_signals

    @property
    def num_bags(self) -> int:
        return self.__num_bags

    @property
    def num_instances(self) -> int:
        return self.__num_instances

    @property
    def num_bits(self) -> int:
        return self.__num_bits

    @property
    def num_targets_high(self) -> int:
        return self.__num_targets

    @property
    def num_targets_low(self) -> int:
        return self.__num_bags - self.__num_targets

    @property
    def num_signals(self) -> int:
        return self.__num_signals

    @property
    def num_signals_per_bag(self) -> int:
        return self.__num_signals_per_bag

    @property
    def initial_seed(self) -> int:
        return self.__seed

    @property
    def bags(self) -> torch.Tensor:
        return self.__data.clone()

    @property
    def targets(self) -> torch.Tensor:
        return self.__targets.clone()

    @property
    def signals(self) -> torch.Tensor:
        return self.__signals.clone()

class FridayDataset(Dataset):
    def __init__(self, data, target) -> None:
        super().__init__()
        assert len(data) == len(target)
        self.data = data
        self.target = target
    def from_dataframe(df, labels) -> None:
        super().__init__()
        data = df[labels[0]]
        target = df[labels[1]]
        return FridayDataset(data, target)
    def from_hf(input, labels, ifppt=None, ifppd = None):
        data = input[labels[0]]
        target = input[labels[1]]
        if ifppt != None:
            data = ifppt(data)
        if ifppd != None:
            target = ifppd(target)
        return FridayDataset(data, target)
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        return {r'data': self.data[idx],  r'target': self.target[idx]}