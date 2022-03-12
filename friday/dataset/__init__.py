from torch.utils.data import Dataset

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