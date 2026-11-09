#pragma once
#include <cstdint>

#include "types.h"

/*
 * RingBuffer<N>
 * A fixed-size circular buffer that holds the N most recent sensor readings.
 *
 * Think of it like a shift register: when it's full and you push a new value,
 * the oldest value is automatically overwritten. This lets us keep a rolling
 * window of recent samples without ever allocating or freeing memory.
 *
 * N is set at compile time (e.g. RingBuffer<30>) so the memory lives on the
 * stack — important on a microcontroller with limited RAM.
 */
template <uint16_t N>
class RingBuffer {
   public:
    // Add a new reading to the buffer.
    // If the buffer is already full, the oldest reading is overwritten.
    void push(const Reading& r);

    // Returns true if the buffer has N readings in it.
    bool full() const { return _count == N; }

    // Returns how many readings are currently stored (0 to N).
    uint16_t count() const { return _count; }

    // Empties the buffer. Does not zero the memory, just resets the counters.
    void clear() {
        _head = 0;
        _count = 0;
    }

    // Access readings in time order: index 0 = oldest, index count-1 = newest.
    // Use this to iterate over the window for feature extraction.
    const Reading& operator[](uint16_t i) const;

   private:
    Reading _buf[N];      // raw storage — fixed array, no heap allocation
    uint16_t _head = 0;   // index where the NEXT write will go
    uint16_t _count = 0;  // how many valid readings are currently stored
};

// --- Implementation (must be in the header because this is a C++ template) ---

template <uint16_t N>
void RingBuffer<N>::push(const Reading& r) {
    _buf[_head] = r;           // write at current head position
    _head = (_head + 1) % N;   // advance head, wrapping around at N
    if (_count < N) _count++;  // only increment count until buffer is full
}

template <uint16_t N>
const Reading& RingBuffer<N>::operator[](uint16_t i) const {
    // When the buffer is full, _head points to the oldest slot (it's about to
    // be overwritten next). When not full, slot 0 is the oldest.
    uint16_t oldest = (_count == N) ? _head : 0;
    return _buf[(oldest + i) % N];
}
