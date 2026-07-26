from __future__ import annotations
from collections.abc import MutableSequence
from typing import TypeVar, Iterator, cast, overload

T = TypeVar('T')


class SequenceView(MutableSequence[T]):
    """
    A view for list slice

    Useful for creating a slice of list but modifying the slice will modify the original list.
    And vice versa.

    Mutating operations (append/insert/pop/remove/etc. — provided by the
    ``MutableSequence`` mixin on top of ``insert`` and ``__delitem__``) write
    through to the parent sequence, matching Pine's ``array.slice`` semantics
    where pushing to or removing from a slice also grows or shrinks the original
    array. ``array.slice`` always builds a contiguous, forward (step 1) range.
    """

    __slots__ = ('sequence', 'range')

    def __init__(self, sequence: MutableSequence[T], range_object: range | None = None) -> None:
        self.range: range = range_object if range_object is not None else range(len(sequence))
        self.sequence = sequence

    @overload
    def __getitem__(self, key: int) -> T: ...

    @overload
    def __getitem__(self, key: slice) -> SequenceView[T]: ...

    def __getitem__(self, key: int | slice) -> T | SequenceView[T]:
        if isinstance(key, slice):
            return SequenceView(self.sequence, self.range[key])
        else:
            return self.sequence[self.range[key]]

    def __setitem__(self, key: int | slice, value: T) -> None:
        if isinstance(key, slice):
            for i in self.range[key]:
                self.sequence[i] = value
        else:
            self.sequence[self.range[key]] = value

    def __delitem__(self, key: int | slice) -> None:
        if isinstance(key, slice):
            # Delete from the parent at the mapped indices, highest first so the
            # earlier indices stay valid; then shrink this view's range.
            removed = 0
            for i in sorted(self.range[key], reverse=True):
                del self.sequence[i]
                removed += 1
            self.range = range(self.range.start, self.range.stop - removed)
        else:
            n = len(self.range)
            if key < 0:
                key += n
            if not 0 <= key < n:
                raise IndexError("SequenceView index out of range")
            del self.sequence[self.range[key]]
            self.range = range(self.range.start, self.range.stop - 1)

    def insert(self, index: int, value: T) -> None:
        # Insert into the parent at the mapped position and extend this view's
        # range by one. Index is clamped like list.insert (append maps to the
        # slice end, i.e. parent index range.stop — Pine's array.push on a slice).
        n = len(self.range)
        if index < 0:
            index += n
        index = max(0, min(index, n))
        self.sequence.insert(self.range.start + index, value)
        self.range = range(self.range.start, self.range.stop + 1)

    def __len__(self) -> int:
        return len(self.range)

    def __iter__(self) -> Iterator[T]:
        for i in self.range:
            yield self.sequence[i]

    def __repr__(self) -> str:
        return f"SequenceView({self.sequence!r}, {self.range!r})"

    def __str__(self) -> str:
        if isinstance(self.sequence, str):
            return ''.join(cast('Iterator[str]', self))
        elif isinstance(self.sequence, (list, tuple)):
            return str(type(self.sequence)(self))
        else:
            return repr(self)
