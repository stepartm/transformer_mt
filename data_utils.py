from typing import Tuple

import torch

class MTData(torch.utils.data.Dataset):

    def __init__(self, source_path: str, target_path: str, lowercase=False):
        self.source_data = open(source_path).readlines()
        self.target_data = open(target_path).readlines()
        self.lowercase = lowercase

    def __getitem__(self, index: int) -> Tuple[str, str]:
        source = self.source_data[index].strip()
        target = self.target_data[index].strip()

        if self.lowercase:
            source = source.lower()
            target = target.lower()

        return (source, target)

    def __len__(self) -> int:
        return len(self.source_data)