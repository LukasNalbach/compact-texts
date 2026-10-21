/**
 * part of LukasNalbach/compact-texts
 *
 * MIT License
 *
 * Copyright (c) Lukas Nalbach
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef COMPACT_TEXTS_COMMON_HPP
#define COMPACT_TEXTS_COMMON_HPP

#include <omp.h>

#include <cstdint>
#include <cstring>

#include <misc/hashing.hpp>

namespace compact_texts {

// returns the 64-bit word stored at data (any alignment)
inline uint64_t load_u64(const void* data) {
  uint64_t value;
  std::memcpy(&value, data, sizeof(uint64_t));
  return value;
}

// mixes the word w into the hash value h
inline uint64_t hash_mix(uint64_t h, uint64_t w) {
  h = (h ^ w) * golden_ratio_64;
  return h ^ (h >> 32);
}

// hashes the words word_at(0), ..., word_at(num_words - 1) in four independent lanes
template <typename word_at_t>
inline uint64_t hash_words(uint64_t num_words, word_at_t word_at) {
  // the first four words of the fraction of pi
  uint64_t a = 0x243F6A8885A308D3ull;
  uint64_t b = 0x13198A2E03707344ull;
  uint64_t c = 0xA4093822299F31D0ull;
  uint64_t d = 0x082EFA98EC4E6C89ull;
  uint64_t k = 0;

  for (; k + 4 <= num_words; k += 4) {
    a = hash_mix(a, word_at(k));
    b = hash_mix(b, word_at(k + 1));
    c = hash_mix(c, word_at(k + 2));
    d = hash_mix(d, word_at(k + 3));
  }

  for (; k < num_words; ++k) a = hash_mix(a, word_at(k));
  return hash_mix(hash_mix(hash_mix(a, b), c), d ^ num_words);
}

// sets the bytes [data, data + bytes) to value with the given number of threads
inline void parallel_memset(void* data, int value, uint64_t bytes, int threads = omp_get_max_threads()) {
#pragma omp parallel num_threads(threads)
  {
    const uint64_t t = omp_get_thread_num();
    const uint64_t nt = omp_get_num_threads();
    const uint64_t beg = bytes * t / nt;
    const uint64_t end = bytes * (t + 1) / nt;
    std::memset(static_cast<uint8_t*>(data) + beg, value, end - beg);
  }
}

}

#endif
