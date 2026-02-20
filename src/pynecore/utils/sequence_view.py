from __future__ import annotations
from typing import TypeVar, Generic, MutableSequence, Iterator

T = TypeVar('T')


class SequenceView(Generic[T]):
    """
    A view for list slice

    Useful for creating a slice of list but modifying the slice will modify the original list.
    And vice versa.
    """

    __slots__ = ('sequence', 'range')

    def __init__(self, sequence: MutableSequence[T], range_object: range | None = None) -> None:
        if range_object is None:
            range_object = range(len(sequence))
        self.range = range_object
        self.sequence = sequence

    def __getitem__(self, key: int | slice) -> T | SequenceView[T]:
        if isinstance(key, slice):
            return SequenceView(self.sequence, self.range[key])
        else:
            return self.sequence[self.range[key]]

    def __setitem__(self, key: int | slice, value: T) -> None:
        self.sequence[self.range[key]] = value  # type: ignore

    def __len__(self) -> int:
        return len(self.range)

    def __iter__(self) -> Iterator[T]:
        for i in self.range:
            if i < len(self.sequence):
                yield self.sequence[i]

    def append(self, value: T) -> None:
        """Append value — inserts into underlying sequence after the view's range and extends the view."""
        insert_pos = self.range.stop
        self.sequence.insert(insert_pos, value)
        self.range = range(self.range.start, self.range.stop + 1, self.range.step)

    def index(self, value: T) -> int:
        """Return index of first occurrence of value within the view."""
        for i, idx in enumerate(self.range):
            if idx < len(self.sequence):
                if self.sequence[idx] == value:
                    return i
        raise ValueError(f"{value!r} is not in SequenceView")

    def insert(self, idx: int, value: T) -> None:
        """Insert value at index within the view."""
        real_idx = self.range[idx] if idx < len(self.range) else self.range.stop
        self.sequence.insert(real_idx, value)
        self.range = range(self.range.start, self.range.stop + 1, self.range.step)

    def pop(self, idx: int = -1) -> T:
        """Remove and return element at index within the view."""
        real_idx = self.range[idx]
        val = self.sequence.pop(real_idx)
        self.range = range(self.range.start, self.range.stop - 1, self.range.step)
        return val

    def clear(self) -> None:
        """Remove all elements in the view from the underlying sequence."""
        for idx in sorted(self.range, reverse=True):
            del self.sequence[idx]
        self.range = range(self.range.start, self.range.start)

    def reverse(self) -> None:
        """Reverse elements within the view."""
        indices = list(self.range)
        values = [self.sequence[i] for i in indices]
        values.reverse()
        for i, idx in enumerate(indices):
            self.sequence[idx] = values[i]

    def sort(self, *, key=None, reverse=False) -> None:
        """Sort elements within the view."""
        indices = list(self.range)
        values = [self.sequence[i] for i in indices]
        values.sort(key=key, reverse=reverse)
        for i, idx in enumerate(indices):
            self.sequence[idx] = values[i]

    def __repr__(self) -> str:
        return f"SequenceView({self.sequence!r}, {self.range!r})"

    def __str__(self) -> str:
        if isinstance(self.sequence, str):
            return ''.join(self)  # type: ignore
        elif isinstance(self.sequence, (list, tuple)):
            return str(type(self.sequence)(self))
        else:
            return repr(self)
