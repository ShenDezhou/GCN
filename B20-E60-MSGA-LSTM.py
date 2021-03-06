from __future__ import absolute_import

import os
import time

from . import (LockBase, NotLocked, NotMyLock, LockTimeout,
               AlreadyLocked)


class SymlinkLockFile(LockBase):
    """Lock access to a file using symlink(2)."""

    def __init__(self, path, threaded=True, timeout=None):
        # super(SymlinkLockFile).__init(...)
        LockBase.__init__(self, path, threaded, timeout)
        # split it back!
        self.unique_name = os.path.split(self.unique_name)[1]

    def acquire(self, timeout=None):
        # Hopefully unnecessary for symlink.
        # try:
        #     open(self.unique_name, "wb").close()
        # except IOError:
        #     raise LockFailed("failed to create %s" % self.unique_name)
        timeout = timeout if timeout is not None else self.timeout
        end_time = time.time()
        if timeout is not None and timeout > 0:
            end_time += timeout

        while True:
            # Try and create a symbolic link to it.
            try:
                os.symlink(self.unique_name, self.lock_file)
            except OSError:
                # Link creation failed.  Maybe we've double-locked?
                if self.i_am_locking():
                    # Linked to out unique name. Proceed.
                    return
                else:
                    # Otherwise the lock creation failed.
                    if timeout is not None and time.time() > end_time:
                        if timeout > 0:
                            raise LockTimeout("Timeout waiting to acquire"
                                              " lock for %s" %
                                              self.path)
                        else:
                            raise AlreadyLocked("%s is already locked" %
                                                self.path)
                    time.sleep(timeout / 10 if timeout is not None else 0.1)
            else:
                # Link creation succeeded.  We're good to go.
                return

    def release(self):
        if not self.is_locked():
            raise NotLocked("%s is not locked" % self.path)
        elif not self.i_am_locking():
            raise NotMyLock("%s is locked, but not by me" % self.path)
        os.unlink(self.lock_file)

    def is_locked(self):
        return os.path.islink(self.lock_file)

    def i_am_locking(self):
        return (os.path.islink(self.lock_file)
                and os.readlink(self.lock_file) == self.unique_name)

    def break_lock(self):
        if os.path.islink(self.lock_file):  # exists && link
            os.unlink(self.lock_file)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                        <listcomp>�	__index__z+Don't know how to index an OrderedSet by %rN)
r   �slice�	SLICE_ALL�copyr   r   r   �list�	__class__�	TypeError)r   �index�resultr   )r   r   �__getitem__F   s    


zOrderedSet.__getitem__c             C   s
   | j | �S )z�
        Return a shallow copy of this object.

        Example:
            >>> this = OrderedSet([1, 2, 3])
            >>> other = this.copy()
            >>> this == other
            True
            >>> this is other
            False
        )r   )r   r   r   r   r   e   s    zOrderedSet.copyc             C   s   t | �dkrdS t| �S d S )Nr   )N)r   r   )r   r   r   r   �__getstate__s   s    zOrderedSet.__getstate__c             C   s"   |dkr| j g � n
| j |� d S )N)N)r   )r   �stater   r   r   �__setstate__   s    zOrderedSet.__setstate__c             C   s
   || j kS )z�
        Test if the item is in this ordered set

        Example:
            >>> 1 in OrderedSet([1, 3, 2])
            True
            >>> 5 in OrderedSet([1, 3, 2])
            False
        )r   )r   �keyr   r   r   �__contains__�   s    
zOrderedSet.__contains__c             C   s0   || j kr&t| j�| j |< | jj|� | j | S )aE  
        Add `key` as an item to this OrderedSet, then return its index.

        If `key` is already in the OrderedSet, return the index it already
        had.

        Example:
            >>> oset = OrderedSet()
            >>> oset.append(3)
            0
            >>> print(oset)
            OrderedSet([3])
        )r   r   r   �append)r   r&   r   r   r   �add�   s    
zOrderedSet.addc             C   sJ   d}yx|D ]}| j |�}qW W n$ tk
rD   tdt|� ��Y nX |S )a<  
        Update the set with the given iterable sequence, then return the index
        of the last element inserted.

        Example:
            >>> oset = OrderedSet([1, 2, 3])
            >>> oset.update([3, 1, 5, 1, 4])
            4
            >>> print(oset)
            OrderedSet([1, 2, 3, 5, 4])
        Nz(Argument needs to be an iterable, got %s)r)   r   �
ValueError�type)r   �sequenceZ
item_index�itemr   r   r   �update�   s    
zOrderedSet.updatec                s$   t |�r� fdd�|D �S � j| S )aH  
        Get the index of a given entry, raising an IndexError if it's not
        present.

        `key` can be an iterable of entries that is not a string, in which case
        this returns a list of indices.

        Example:
            >>> oset = OrderedSet([1, 2, 3])
            >>> oset.index(2)
            1
        c                s   g | ]}� j |��qS r   )r    )r   Zsubkey)r   r   r   r   �   s    z$OrderedSet.index.<locals>.<listcomp>)r   r   )r   r&   r   )r   r   r    �   s    zOrderedSet.indexc             C   s,   | j std�